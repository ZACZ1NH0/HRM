from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

from transformers import AutoTokenizer
try:
    from reader_baseline import QAPipeline  # file bạn đưa
except Exception:
    QAPipeline = None  # phòng trường hợp chưa có file

@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len_q: int
    ctx_k: int
    ctx_len: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    tokenizer_name: Optional[str] = None
    use_external_reader: bool = False
    answer_max_len: int = 32


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.total_len = self.config.seq_len_q + self.config.ctx_k * self.config.ctx_len
        # I/O
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight = self.embed_tokens.embedding_weight
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.segment_embed = nn.Parameter(torch.zeros(2, self.config.hidden_size, dtype=self.forward_dtype))
        # self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        # if self.config.puzzle_emb_ndim > 0:
        #     # Zero init puzzle embeddings
        #     self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
        #                                             batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.total_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.total_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.H_layers)])
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])
        
        _H_init = trunc_normal_init_(torch.empty(1, 1, self.config.hidden_size, dtype=self.forward_dtype), std=1)
        _L_init = trunc_normal_init_(torch.empty(1, 1, self.config.hidden_size, dtype=self.forward_dtype), std=1)

        # Đăng ký buffer để .to(device) hoạt động
        self.register_buffer("H_init", _H_init, persistent=True)
        self.register_buffer("L_init", _L_init, persistent=True)
        # Initial states
        # self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        # self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

        self.reader = None
        self._tok = None
        if getattr(self.config, "use_external_reader", False):
            assert self.config.tokenizer_name is not None, "tokenizer_name phải được set trong config khi dùng external reader"
            self._tok = AutoTokenizer.from_pretrained(self.config.tokenizer_name, use_fast=True)
            if QAPipeline is None:
                raise RuntimeError("Không tìm thấy QAPipeline (reader_baseline.py). Hãy đặt file vào PYTHONPATH.")
            self.reader = QAPipeline()

    def _input_embeddings(self, question_ids: torch.Tensor, ctx_ids: torch.Tensor):
        """
        question_ids: (B, Lq)
        ctx_ids:      (B, K, Lc)
        return:       (B, total_len, H)
        """
        B, K, Lc = ctx_ids.shape

        q_emb   = self.embed_tokens(question_ids.to(torch.int32))             # (B, Lq, H)
        ctx_emb = self.embed_tokens(ctx_ids.view(B, K*Lc).to(torch.int32))    # (B, K*Lc, H)

        seg = self.segment_embed  # (2, H)
        ctx_emb = ctx_emb + seg[0]   # CTX
        q_emb   = q_emb   + seg[1]   # Q

        emb = torch.cat([ctx_emb, q_emb], dim=1)  # (B, total_len, H)

        if self.config.pos_encodings == "learned":
            emb = 0.707106781 * (emb + self.embed_pos.embedding_weight.to(self.forward_dtype))
        return self.embed_scale * emb


    # def empty_carry(self, batch_size: int):
    #     return HierarchicalReasoningModel_ACTV1InnerCarry(
    #         z_H=torch.empty(batch_size, self.total_len, self.config.hidden_size, dtype=self.forward_dtype),
    #         z_L=torch.empty(batch_size, self.total_len, self.config.hidden_size, dtype=self.forward_dtype),
    # )
    def empty_carry(self, batch_size: int):
        total_len = self.config.seq_len_q + self.config.ctx_k * self.config.ctx_len
        device = self.segment_embed.device  # bám theo tham số của model
        dtype  = self.forward_dtype
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, total_len, self.config.hidden_size, dtype=dtype, device=device),
            z_L=torch.empty(batch_size, total_len, self.config.hidden_size, dtype=dtype, device=device),
        )


        
    # def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
    #     return HierarchicalReasoningModel_ACTV1InnerCarry(
    #         z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
    #         z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
    #     )
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        # reset_flag: (B,)
        B = reset_flag.shape[0]
        # broadcast H_init/L_init từ (1,1,H) → (B, seq_len, H)
        H_init = self.H_init.expand(B, carry.z_H.shape[1], -1)
        L_init = self.L_init.expand(B, carry.z_L.shape[1], -1)

        mask = reset_flag.view(-1, 1, 1).to(carry.z_H.device)
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(mask, H_init, carry.z_H),
            z_L=torch.where(mask, L_init, carry.z_L),
        )
    
    def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["ctx_inputs"])

        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

                # >>> THÊM: nếu dùng external reader và đang EVAL, thay lm_head bằng reader pretrained
        if (self.reader is not None) and (not self.training):
            B, total_len, V = z_H.shape[0], z_H.shape[1], self.config.vocab_size
            device = z_H.device
            # logits "giả lập": -inf ở mọi vị trí, riêng tail ghi one-hot theo answer tokens
            output = torch.full((B, total_len, V), -1e9, dtype=torch.float32, device=device)

            def _ids_to_text(ids_1d):
                if isinstance(ids_1d, torch.Tensor):
                    ids_1d = ids_1d.tolist()
                return self._tok.decode(ids_1d, skip_special_tokens=True).strip()

            # lấy question/context text từ batch ids
            B_, K, Lc = batch["ctx_inputs"].shape
            Lq = batch["inputs"].shape[1]
            assert B_ == B

            for i in range(B):
                q_text = _ids_to_text(batch["inputs"][i])
                ctx_texts = []
                for k in range(K):
                    ctx_texts.append(_ids_to_text(batch["ctx_inputs"][i, k]))
                context = " ".join(ctx_texts)

                # gọi reader pretrained
                pred_text, _score = self.reader.answer(q_text, context, max_len=384)

                # mã hóa lại thành token ids (không thêm special)
                ans_ids = self._tok(pred_text, add_special_tokens=False)["input_ids"][: self.config.answer_max_len]
                L = len(ans_ids)
                if L > 0:
                    start = total_len - L  # dán answer vào đuôi như labels đang làm
                    for t, tid in enumerate(ans_ids):
                        if 0 <= tid < V and 0 <= start + t < total_len:
                            # đặt logit lớn để argmax ra đúng token pred
                            output[i, start + t, tid] = 50.0

            # q_head vẫn tính như cũ
            q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
            new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
            return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        device = next(self.parameters()).device
        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=device),  # Default to halted
            
            current_data={k: torch.empty_like(v, device = device) for k, v in batch.items()}
        )
        
    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                
                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
