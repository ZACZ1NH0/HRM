from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from retriever import BM25Retriever
import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2 import AdamATan2
from transformers import AutoTokenizer
import re, string
from hotpot_dataset import HotpotQADataset, HotpotQADatasetConfig, HotpotQADatasetMetadata
from utils.functions import load_model_class, get_model_source_path

os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")




class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    loss: LossConfig


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    data_path: str  # không dùng nữa cho HF datasets, nhưng giữ cho đủ chữ ký

    # Hyperparams
    global_batch_size: int
    epochs: int
    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float

    # --- THÊM: QA lengths + tokenizer ---
    seq_len_q: int
    ctx_k: int
    ctx_len: int
    tokenizer_name: str = "bert-base-uncased"

    # (bỏ các trường puzzle_emb_*)
    # project/run names, seed, ...
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = []


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    # kwargs có thể bỏ qua; rank/world_size giữ cho đủ chữ ký
    retr = BM25Retriever()
    cache_path = "artifacts/bm25_fullwiki.pkl"
    if os.path.exists(cache_path):
        retr.load(cache_path, show_stats=True)
    else:
        raise FileNotFoundError(
            f"BM25 cache not found at {cache_path}. Build fullwiki index first and save it."
        )
    ds_cfg = HotpotQADatasetConfig(
        tokenizer_name=config.tokenizer_name,    # thêm field này vào PretrainConfig bên dưới
        seq_len_q=config.seq_len_q,
        ctx_k=config.ctx_k,
        ctx_len=config.ctx_len,
        seed=config.seed,
        use_supporting_facts=False,
        global_batch_size=config.global_batch_size,
        retriever=retr,
    )
    
    dataset = HotpotQADataset("train" if split=="train" else "validation", ds_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle = False,     # giữ nguyên: dataset trả triple
        num_workers=2,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: HotpotQADatasetMetadata, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,

        # QA lengths
        vocab_size=train_metadata.vocab_size,
        seq_len_q=train_metadata.seq_len_q,
        ctx_k=train_metadata.ctx_k,
        ctx_len=train_metadata.ctx_len,
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        # if "DISABLE_COMPILE" not in os.environ:
        #     model = torch.compile(model, dynamic=False)  # type: ignore
        if os.environ.get("DISABLE_COMPILE", "1") != "1":  # mặc định TẮT compile
            model = torch.compile(model, dynamic=False)
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # chỉ còn 1 optimizer cho toàn model (không còn puzzle emb)
    optimizers = [
        AdamATan2(
            model.parameters(),
            lr=0,  # set bởi scheduler
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2)
        )
    ]
    optimizer_lrs = [config.lr]
    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: HotpotQADatasetMetadata, world_size: int):
    # ước lượng: mỗi step tiêu thụ ~ global_batch_size items
    total_steps = int(config.epochs * train_metadata.total_groups / config.global_batch_size)

    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )


def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()
    
    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
    torch.nn.utils.clip_grad_norm_(train_state.model.parameters(), 1.0)        
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics


def _normalize_text(s: str) -> str:
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def _f1_score(pred: str, gold: str) -> float:
    pred_toks = _normalize_text(pred).split()
    gold_toks = _normalize_text(gold).split()
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return float(pred_toks == gold_toks)
    common = set(pred_toks) & set(gold_toks)
    num_same = sum(min(pred_toks.count(t), gold_toks.count(t)) for t in common)
    if num_same == 0: return 0.0
    precision = num_same / len(pred_toks)
    recall    = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)

def _exact_match(pred: str, gold: str) -> float:
    return float(_normalize_text(pred) == _normalize_text(gold))

def evaluate(config, train_state, eval_loader, eval_metadata, rank: int, world_size: int):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)
    # stop_ids = {tokenizer.sep_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
    stop_ids = {i for i in [tokenizer.sep_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id] if i is not None}
    answer_max_len = getattr(config, "answer_max_len", 32)   # hoặc để 32 mặc định

    em_sum = 0.0
    f1_sum = 0.0
    n_examples = 0

    with torch.no_grad():
        for _set_name, batch, _gbs in eval_loader:
            # === gọi model để lấy logits ===
            # LƯU Ý: head (ACTLossHead) phải forward(..., return_keys=["logits"]) để có out_dict["logits"]
            batch = {k: v.cuda() for k, v in batch.items()}
            carry = train_state.model.model.initial_carry(batch)  # nếu bạn đã có carry trước đó thì dùng lại
            carry, loss, metrics, out_dict, extras = train_state.model(
                carry=carry,
                batch=batch,
                return_keys=["logits"]   # <-- quan trọng
            )

            logits = out_dict["logits"]              # (B, total_len, vocab)
            labels = batch["labels"]                 # (B, total_len) với -100 ngoài vùng answer
            B, total_len, _ = logits.shape

            # === Lấy phần tail cho câu trả lời ===
            tail_logits = logits[:, -answer_max_len:, :]         # (B, Lans, vocab)
            pred_ids = tail_logits.argmax(-1).cpu().tolist()     # List[List[int]]

            # === Decode prediction (dừng sớm ở SEP/EOS/PAD) ===
            pred_texts = []
            for ids in pred_ids:
                toks = []
                for t in ids:
                    if t in stop_ids: break
                    toks.append(t)
                pred_texts.append(tokenizer.decode(toks, skip_special_tokens=True).strip())

            # === Lấy gold text từ labels (bỏ -100) ===
            gold_texts = []
            for lab in labels.cpu().tolist():
                ans_tok = [t for t in lab if t != -100]
                gold_texts.append(tokenizer.decode(ans_tok, skip_special_tokens=True).strip())

            # === Tính EM/F1 batch ===
            for p, g in zip(pred_texts, gold_texts):
                em_sum += _exact_match(p, g)
                f1_sum += _f1_score(p, g)
                n_examples += 1

    # Trung bình
    if n_examples > 0:
        em_avg = em_sum / n_examples
        f1_avg = f1_sum / n_examples
    else:
        em_avg = 0.0
        f1_avg = 0.0

    # gộp vào metrics để log W&B
    metrics = {
        "eval/em": em_avg,
        "eval/f1": f1_avg,
        # có thể thêm loss eval hiện có nếu bạn đã tính
    }
    return metrics

# def evaluate(config: PretrainConfig, train_state: TrainState, eval_loader: torch.utils.data.DataLoader, eval_metadata: HotpotQADatasetMetadata, rank: int, world_size: int):
#     with torch.inference_mode():
#         set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
        
#         all_preds = {}

#         metric_keys = []
#         metric_values = None
#         metric_global_batch_size = [0 for _ in range(len(set_ids))]
        
#         carry = None
#         for set_name, batch, global_batch_size in eval_loader:
#             # To device
#             batch = {k: v.cuda() for k, v in batch.items()}
#             with torch.device("cuda"):
#                 carry = train_state.model.initial_carry(batch)  # type: ignore

#             # Forward
#             while True:
#                 carry, _, metrics, preds, all_finish = train_state.model(carry=carry, batch=batch, return_keys=config.eval_save_outputs)
                
#                 if all_finish:
#                     break

#             for collection in (batch, preds):
#                 for k, v in collection.items():
#                     if k in config.eval_save_outputs:
#                         all_preds.setdefault(k, [])
#                         all_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory
                        
#             del carry, preds, batch, all_finish

#             # Aggregate
#             set_id = set_ids[set_name]
            
#             if metric_values is None:
#                 metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
#                 metric_values = torch.zeros((len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda")
                
#             metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
#             metric_global_batch_size[set_id] += global_batch_size

#         if len(all_preds) and config.checkpoint_path is not None:
#             all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}

#             os.makedirs(config.checkpoint_path, exist_ok=True)
#             torch.save(all_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"))

#         # Logging
#         # Reduce to rank 0
#         if metric_values is not None:
#             if world_size > 1:
#                 dist.reduce(metric_values, dst=0)
            
#             if rank == 0:
#                 reduced_metrics = metric_values.cpu().numpy()
#                 reduced_metrics = {set_name: {metric_name: reduced_metrics[set_id, metric_id] for metric_id, metric_name in enumerate(metric_keys)}
#                                    for set_id, set_name in enumerate(set_ids)}
                
#                 # Postprocess
#                 for set_name, metrics in reduced_metrics.items():
#                     count = metrics.pop("count")
#                     reduced_metrics[set_name] = {k: v / count for k, v in metrics.items()}

#                 return reduced_metrics


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_path).capitalize()} ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)
    # max_steps_per_epoch = getattr(config, "max_steps_per_epoch", None)
    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)

    # Train state
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)

        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)


    #Training Loop
    # for _iter_id in range(total_iters):
    #     for local_epoch in range(train_epochs_per_iter):
    #         epoch_idx = _iter_id * train_epochs_per_iter + local_epoch
    #         print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {epoch_idx}")

    #         ############ Train Iter (1 epoch)
    #         train_state.model.train()
    #         # đếm step trong epoch
    #         steps_in_epoch = 0
    #         for set_name, batch, global_batch_size in train_loader:
    #             metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

    #             steps_in_epoch += 1
    #             if RANK == 0 and metrics is not None:
    #                 wandb.log(metrics, step=train_state.step)
    #                 progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
    #                 if steps_in_epoch % 200 == 0:
    #                     print(f"[Epoch {epoch_idx}] step {steps_in_epoch}")

    #             # GIỚI HẠN SỐ BƯỚC MỖI EPOCH (tùy chọn)
    #             if (max_steps_per_epoch is not None) and (steps_in_epoch >= max_steps_per_epoch):
    #                 print(f"[Epoch {epoch_idx}] capped at {max_steps_per_epoch} steps")
    #                 break  # kết thúc sớm epoch này

    #         ############ Evaluation (sau mỗi epoch)
    #         train_state.model.eval()
    #         metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)
    #         if RANK == 0 and metrics is not None:
    #             wandb.log(metrics, step=train_state.step)

    #         ############ Checkpointing (sau mỗi epoch)
    #         if RANK == 0 and config.checkpoint_every_eval:
    #             save_train_state(config, train_state)

    # # finalize
    # if dist.is_initialized():
    #     dist.destroy_process_group()
    # wandb.finish()
    # Training Loop
    for _iter_id in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore

        ############ Evaluation
        train_state.model.eval()
        metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

        if RANK == 0 and metrics is not None:
            wandb.log(metrics, step=train_state.step)
            
        ############ Checkpointing
        if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
            save_train_state(config, train_state)

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
