# hotpot_dataset.py
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import random
import math

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from retriever import BM25Retriever as retriever


@dataclass
class HotpotQADatasetConfig:
    # tokenization / geometry
    tokenizer_name: str = "bert-base-uncased"
    seq_len_q: int = 64       # Lq
    ctx_k: int = 4            # K passages
    ctx_len: int = 128        # Lc
    retriever: Optional[object] = None
    # misc
    seed: int = 0
    use_supporting_facts: bool = True  # True = dùng supporting_facts để lấy passages
    global_batch_size: int = 4         # để trả về ở triple (giữ nguyên giao diện pretrain.py)


@dataclass
class HotpotQADatasetMetadata:
    # những trường pretrain.py cần tới
    vocab_size: int
    # các trường dưới đây thay thế logic puzzle cho ước lượng steps/eval
    total_groups: int
    mean_examples_per_group: int
    sets: List[str]                   # ví dụ: ["train"] hoặc ["validation"]

    # QA-specific (để create_model)
    seq_len_q: int
    ctx_k: int
    ctx_len: int


class HotpotQADataset(Dataset):
    """
    Trả về: (set_name, batch_dict, global_batch_size)
    Trong đó batch_dict gồm:
        inputs:     (Lq,)             # question ids
        ctx_inputs: (K, Lc)           # K passages đã token hóa
        labels:     (K*Lc + Lq,)      # -100 ngoài vùng answer
    """
    def __init__(self, split: str, cfg: HotpotQADatasetConfig):
        super().__init__()
        assert split in ("train", "validation")
        self.split = split
        self.cfg = cfg
        self.retriever = cfg.retriever

        # tokenizer
        self.tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
        if self.tok.pad_token_id is None:
            # đảm bảo có PAD
            self.tok.pad_token = self.tok.eos_token if self.tok.eos_token else "[PAD]"

        # load dataset (chọn "distractor" để nhanh; "fullwiki" cũng được nếu bạn đã login)
        self.ds = load_dataset("hotpot_qa", "fullwiki")[split]
        self.num_groups = math.ceil(len(self.ds) / cfg.global_batch_size)

        random.seed(cfg.seed)

        # metadata để trả về từ create_dataloader
        self.metadata = HotpotQADatasetMetadata(
            vocab_size=self.tok.vocab_size,
            total_groups=len(self.ds),                # mỗi item là 1 "group"
            mean_examples_per_group=1,                # đơn giản hóa
            sets=[split],
            seq_len_q=cfg.seq_len_q,
            ctx_k=cfg.ctx_k,
            ctx_len=cfg.ctx_len
        )

    def _retrieve_passages(self, question_text: str, ex: Dict) -> List[str]:
        """
        Strategy:
          - nếu use_supporting_facts=True: lấy từ supporting_facts (warm-up, đảm bảo vàng có mặt)
          - nếu tắt: yêu cầu retriever instance (BM25, dense) để lấy top-K
        """
        # ưu tiên supporting_facts khi bật để model học dễ hơn
        if self.cfg.use_supporting_facts:
            passages = self._passages_from_supporting(ex)
            if passages:
                return passages[: self.cfg.ctx_k]

        # nếu không có hoặc tắt, dùng retriever
        if self.retriever is None:
            # fallback: lấy ngẫu nhiên từ context distractor (đỡ crash)
            ctx_paras = self._all_paragraphs_from_context(ex)
            random.shuffle(ctx_paras)
            return ctx_paras[: self.cfg.ctx_k]

        # dùng retriever (đã build/load ở pretrain.py)
        top = self.retriever.search(question_text, k=self.cfg.ctx_k)  # [(idx, score)]
        idxs = [i for i, _ in top]
        return self.retriever.get_passages(idxs)
    
    def _encode_ctx(self, passages: List[str]) -> List[List[int]]:
        ctx_ids = []
        for p in passages[:self.ctx_k]:
            ids = self.tok(
                p, truncation=True, padding="max_length",
                max_length=self.ctx_len, add_special_tokens=False
            )["input_ids"]
            ctx_ids.append(ids)
        # pad K
        while len(ctx_ids) < self.ctx_k:
            ctx_ids.append([self.tok.pad_token_id] * self.ctx_len)
        return ctx_ids

    def __iter__(self):
        for ex in self.stream:  # cách bạn đang đọc examples
            q_text = ex["question"]
            passages = self._retrieve_passages(q_text)
            ctx_ids = self._encode_ctx(passages)   # (K, Lc)
            q_ids = self.tok(
                q_text, truncation=True, padding="max_length",
                max_length=self.cfg.seq_len_q, add_special_tokens=False
            )["input_ids"]

            # build labels (đặt -100 cho CTX + Q; answer tokens ở đuôi)
            ans_ids = self.tok(
                ex["answer"], truncation=True, padding="max_length",
                max_length=self.cfg.answer_max_len, add_special_tokens=False
            )["input_ids"]

            yield {
                "inputs": torch.tensor(q_ids, dtype=torch.long),             # (Lq)
                "ctx_inputs": torch.tensor(ctx_ids, dtype=torch.long),# (K, Lc)
                "labels": build_labels(q_ids, ctx_ids, ans_ids, self.tok.pad_token_id),
                # … các field khác bạn đang dùng
            }
    # ===== helpers =====
    def _passages_from_supporting(self, ex):
        """
        Hỗ trợ cả 2 schema của HotpotQA:
        - context = [[title, [sentences...]], ...]
        - context = [{"title": ..., "sentences": [...]}, ...]
        """
        title2sents = {}

        for entry in ex["context"]:
            if isinstance(entry, dict):
                title = entry.get("title")
                sents = entry.get("sentences") or entry.get("text") or []
            elif isinstance(entry, (list, tuple)):
                # Chấp nhận list dài >2, chỉ lấy 2 phần đầu
                title = entry[0] if len(entry) >= 1 else None
                sents = entry[1] if len(entry) >= 2 else []
            else:
                title, sents = None, []

            if isinstance(title, str) and isinstance(sents, (list, tuple)):
                # Mỗi phần tử của sents phải là string
                sents = [s for s in sents if isinstance(s, str)]
                title2sents[title] = sents

        # supporting_facts có thể là [["Title", sent_id], ...] hoặc dicts
        by_title = {}
        for sf in ex.get("supporting_facts", []):
            if isinstance(sf, dict):
                t = sf.get("title")
                sid = sf.get("sent_id")
            elif isinstance(sf, (list, tuple)) and len(sf) >= 2:
                t, sid = sf[0], sf[1]
            else:
                t, sid = None, None
            if isinstance(t, str) and isinstance(sid, int):
                by_title.setdefault(t, []).append(sid)

        passages = []
        for t, sids in by_title.items():
            sents = title2sents.get(t, [])
            if not sents:
                continue
            sids = sorted([sid for sid in sids if 0 <= sid < len(sents)])
            if not sids:
                continue
            txt = " ".join(sents[sid] for sid in sids)
            if txt.strip():
                passages.append(txt)

        return passages[: self.cfg.ctx_k]

    def _tokenize_fixed(self, text: str, max_len: int) -> List[int]:
        return self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            add_special_tokens=False
        )["input_ids"]

    # ===== Dataset API =====
    def __len__(self):
        return self.num_groups

    def __getitem__(self, group_idx: int):
        B = self.cfg.global_batch_size
        start = group_idx * B
        end   = min((group_idx + 1) * B, len(self.ds))
        idxs = list(range(start, end))

        inputs_list = []
        ctx_list    = []
        labels_list = []

        for idx in idxs:
            ex = self.ds[idx]
            q = ex["question"]
            a = ex["answer"]

            # question ids (Lq)
            q_ids = self._tokenize_fixed(q, self.cfg.seq_len_q)

            # passages (K x Lc)
            if self.cfg.use_supporting_facts:
                passages = self._passages_from_supporting(ex)
            else:
                passages = []
            if not passages:
                passages = []
            while len(passages) < self.cfg.ctx_k:
                passages.append("")
            ctx_ids = [self._tokenize_fixed(p, self.cfg.ctx_len) for p in passages[: self.cfg.ctx_k]]

            # labels (generative): answer vào cuối chuỗi [CTX... + Q]
            total_len = self.cfg.ctx_k * self.cfg.ctx_len + self.cfg.seq_len_q
            labels = [-100] * total_len
            a_ids = self.tok(a, add_special_tokens=False)["input_ids"]
            a_ids = a_ids[: min(len(a_ids), total_len)]
            start_pos = total_len - len(a_ids)
            labels[start_pos: start_pos + len(a_ids)] = a_ids

            inputs_list.append(torch.tensor(q_ids, dtype=torch.long))        # (Lq,)
            ctx_list.append(torch.tensor(ctx_ids, dtype=torch.long))         # (K, Lc)
            labels_list.append(torch.tensor(labels, dtype=torch.long))       # (total_len,)

        # Nếu nhóm cuối nhỏ hơn B: pad thêm mẫu rỗng để giữ shape ổn định (tuỳ bạn)
        if len(inputs_list) < B:
            pad_q   = torch.full((self.cfg.seq_len_q,), self.tok.pad_token_id, dtype=torch.long)
            pad_ctx = torch.full((self.cfg.ctx_k, self.cfg.ctx_len), self.tok.pad_token_id, dtype=torch.long)
            pad_lab = torch.full((self.cfg.ctx_k * self.cfg.ctx_len + self.cfg.seq_len_q,), -100, dtype=torch.long)
            while len(inputs_list) < B:
                inputs_list.append(pad_q)
                ctx_list.append(pad_ctx)
                labels_list.append(pad_lab)

        batch = {
            "inputs":     torch.stack(inputs_list, dim=0),   # (B, Lq)
            "ctx_inputs": torch.stack(ctx_list,  dim=0),     # (B, K, Lc)
            "labels":     torch.stack(labels_list, dim=0),   # (B, total_len)
        }
        
        return self.split, batch, self.cfg.global_batch_size
