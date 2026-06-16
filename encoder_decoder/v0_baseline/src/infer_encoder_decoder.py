import os
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# 基于脚本文件位置解析项目根目录，确保从任意位置运行均可正确找到 tokenizer。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# 0. 路径与模型配置
# ============================================================

MODEL_DIR = "minimal_transformer_en_zh_opus_outputs"
MODEL_PATH = os.path.join(MODEL_DIR, "checkpoint_epoch_4.pt")

# 优先使用项目内置 tokenizer。
TOKENIZER_DIR = os.path.join(_PROJECT_ROOT, "tokenizer")
FALLBACK_TOKENIZER_DIR = os.path.join(_PROJECT_ROOT, "tokenizer")

# 推理长度配置。
MAX_SRC_LEN = 96
MAX_NEW_TOKENS = 96
MIN_NEW_TOKENS = 1


@dataclass
class ModelConfig:
    d_model: int = 512
    d_ff: int = 2048
    num_heads: int = 8
    num_layers: int = 8
    dropout: float = 0.05
    max_src_len: int = 96
    max_new_tokens: int = 96


PAD_ID = 0
BOS_ID = 101
EOS_ID = 102


def get_best_device() -> torch.device:
    """
    服务器推理优先使用 CUDA；本地调试时可回退 MPS/CPU。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")




# ============================================================
# 1. Mask
# ============================================================

def make_src_mask(src: torch.Tensor, pad_id: int) -> torch.Tensor:
    # src: [B, S]
    # src_mask: [B, 1, 1, S]
    src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)
    return src_mask


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # causal_mask: [1, 1, T, T]
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    return causal_mask


def make_tgt_mask(tgt_in: torch.Tensor, pad_id: int) -> torch.Tensor:
    # tgt_in: [B, T]
    tgt_pad_mask = (tgt_in != pad_id).unsqueeze(1).unsqueeze(2)
    # tgt_pad_mask: [B, 1, 1, T]

    causal_mask = make_causal_mask(tgt_in.size(1), tgt_in.device)
    # causal_mask: [1, 1, T, T]

    tgt_mask = tgt_pad_mask & causal_mask
    # tgt_mask: [B, 1, T, T]

    return tgt_mask


# ============================================================
# 2. Transformer 模型结构
# ============================================================

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_model]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        layer_norm_out = self.gamma * x_hat + self.beta

        return layer_norm_out


class PostNormResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)

    def forward(
        self,
        residual: torch.Tensor,
        sublayer_out: torch.Tensor,
    ) -> torch.Tensor:
        dropped_sublayer_out = self.dropout(sublayer_out)
        residual_out = residual + dropped_sublayer_out
        post_norm_out = self.norm(residual_out)

        return post_norm_out


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, L]
        token_embeddings = self.embedding(token_ids)
        # token_embeddings: [B, L, d_model]

        scaled_embeddings = token_embeddings * math.sqrt(self.d_model)
        return scaled_embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sinusoidal PE."

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        # pe: [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        seq_len = x.size(1)

        pos_encoding = self.pe[:, :seq_len, :]
        # pos_encoding: [1, L, d_model]

        x = x + pos_encoding
        x = self.dropout(x)

        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        hidden = self.linear_1(x)
        # hidden: [B, L, d_ff]

        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)

        ffn_out = self.linear_2(hidden)
        # ffn_out: [B, L, d_model]

        return ffn_out


def attention_core(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> torch.Tensor:
    # query: [B, H, L_q, d_head]
    # key:   [B, H, L_k, d_head]
    # value: [B, H, L_k, d_head]
    d_head = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_head)
    # scores: [B, H, L_q, L_k]

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = torch.softmax(scores, dim=-1)
    # attention_weights: [B, H, L_q, L_k]

    if dropout is not None:
        attention_weights = dropout(attention_weights)

    context = torch.matmul(attention_weights, value)
    # context: [B, H, L_q, d_head]

    return context


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        batch_size, seq_len, _ = x.shape

        x = x.view(batch_size, seq_len, self.num_heads, self.d_head)
        # x: [B, L, H, d_head]

        x = x.transpose(1, 2)
        # x: [B, H, L, d_head]

        return x

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, L, d_head]
        batch_size, _, seq_len, _ = x.shape

        x = x.transpose(1, 2).contiguous()
        # x: [B, L, H, d_head]

        x = x.view(batch_size, seq_len, self.num_heads * self.d_head)
        # x: [B, L, d_model]

        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # query: [B, L_q, d_model]
        # key:   [B, L_k, d_model]
        # value: [B, L_k, d_model]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        context = attention_core(
            query=q,
            key=k,
            value=v,
            mask=mask,
            dropout=self.dropout,
        )

        context = self.combine_heads(context)

        attention_out = self.out_proj(context)
        # attention_out: [B, L_q, d_model]

        return attention_out


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ):
        super().__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward

        self.self_attn_connection = PostNormResidualConnection(d_model, dropout)
        self.ffn_connection = PostNormResidualConnection(d_model, dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, S, d_model]

        self_attn_out = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask=src_mask,
        )
        # self_attn_out: [B, S, d_model]

        x = self.self_attn_connection(
            residual=x,
            sublayer_out=self_attn_out,
        )
        # x: [B, S, d_model]

        ffn_out = self.feed_forward(x)
        # ffn_out: [B, S, d_model]

        x = self.ffn_connection(
            residual=x,
            sublayer_out=ffn_out,
        )
        # x: [B, S, d_model]

        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadAttention,
        cross_attn: MultiHeadAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ):
        super().__init__()

        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward

        self.self_attn_connection = PostNormResidualConnection(d_model, dropout)
        self.cross_attn_connection = PostNormResidualConnection(d_model, dropout)
        self.ffn_connection = PostNormResidualConnection(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # x:      [B, T, d_model]
        # memory: [B, S, d_model]

        self_attn_out = self.self_attn(
            query=x,
            key=x,
            value=x,
            mask=tgt_mask,
        )
        # self_attn_out: [B, T, d_model]

        x = self.self_attn_connection(
            residual=x,
            sublayer_out=self_attn_out,
        )
        # x: [B, T, d_model]

        cross_attn_out = self.cross_attn(
            query=x,
            key=memory,
            value=memory,
            mask=src_mask,
        )
        # cross_attn_out: [B, T, d_model]

        x = self.cross_attn_connection(
            residual=x,
            sublayer_out=cross_attn_out,
        )
        # x: [B, T, d_model]

        ffn_out = self.feed_forward(x)
        # ffn_out: [B, T, d_model]

        x = self.ffn_connection(
            residual=x,
            sublayer_out=ffn_out,
        )
        # x: [B, T, d_model]

        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        for encoder_layer in self.layers:
            x = encoder_layer(x, src_mask)

        memory = x
        return memory


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for decoder_layer in self.layers:
            x = decoder_layer(x, memory, src_mask, tgt_mask)

        decoder_out = x
        return decoder_out


class Generator(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, decoder_out: torch.Tensor) -> torch.Tensor:
        # decoder_out: [B, T, d_model]
        logits = self.proj(decoder_out)
        # logits: [B, T, vocab_size]

        return logits


class EncoderDecoderTransformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Module,
        tgt_embed: nn.Module,
        generator: Generator,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src_embedded = self.src_embed(src)
        # src_embedded: [B, S, d_model]

        memory = self.encoder(src_embedded, src_mask)
        # memory: [B, S, d_model]

        return memory

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_embedded = self.tgt_embed(tgt_in)
        # tgt_embedded: [B, T, d_model]

        decoder_out = self.decoder(tgt_embedded, memory, src_mask, tgt_mask)
        # decoder_out: [B, T, d_model]

        return decoder_out

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        decoder_out = self.decode(tgt_in, memory, src_mask, tgt_mask)

        logits = self.generator(decoder_out)
        # logits: [B, T, tgt_vocab_size]

        return logits


def make_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    num_layers: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    num_heads: int = 8,
    dropout: float = 0.1,
) -> EncoderDecoderTransformer:
    def build_encoder_layer() -> EncoderLayer:
        self_attn = MultiHeadAttention(num_heads, d_model, dropout)
        feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        encoder_layer = EncoderLayer(
            d_model=d_model,
            self_attn=self_attn,
            feed_forward=feed_forward,
            dropout=dropout,
        )

        return encoder_layer

    def build_decoder_layer() -> DecoderLayer:
        self_attn = MultiHeadAttention(num_heads, d_model, dropout)
        cross_attn = MultiHeadAttention(num_heads, d_model, dropout)
        feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        decoder_layer = DecoderLayer(
            d_model=d_model,
            self_attn=self_attn,
            cross_attn=cross_attn,
            feed_forward=feed_forward,
            dropout=dropout,
        )

        return decoder_layer

    encoder_layers = nn.ModuleList(
        [build_encoder_layer() for _ in range(num_layers)]
    )

    decoder_layers = nn.ModuleList(
        [build_decoder_layer() for _ in range(num_layers)]
    )

    src_embed = nn.Sequential(
        TokenEmbedding(src_vocab_size, d_model),
        PositionalEncoding(d_model, dropout),
    )

    tgt_embed = nn.Sequential(
        TokenEmbedding(tgt_vocab_size, d_model),
        PositionalEncoding(d_model, dropout),
    )

    model = EncoderDecoderTransformer(
        encoder=Encoder(encoder_layers),
        decoder=Decoder(decoder_layers),
        src_embed=src_embed,
        tgt_embed=tgt_embed,
        generator=Generator(d_model, tgt_vocab_size),
    )

    return model


# ============================================================
# 3. Tokenizer / Model Loading
# ============================================================

def load_local_tokenizer(tokenizer_dir: str):
    """
    优先用 AutoTokenizer 加载训练时保存的 tokenizer；
    如果本地文件不完整，则回退到 PreTrainedTokenizerFast(tokenizer_file=...).
    """
    if not os.path.isdir(tokenizer_dir):
        raise FileNotFoundError(f"tokenizer dir not found: {tokenizer_dir}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            use_fast=True,
            #优先使用 Rust 实现的 fast tokenizer
            local_files_only=True,
            #表示只从本地加载，不允许联网下载
        )
        return tokenizer
    except Exception as e:
        print(f"AutoTokenizer load failed: {e}")
        print("Fallback to PreTrainedTokenizerFast(tokenizer_file=...).")

        tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")
        # tokenizer.json 是 fast tokenizer 的核心文件；
        # 里面保存了词表、分词规则、normalizer、pre-tokenizer、post-processor 等信息。

        # 绕过 AutoTokenizer 的自动识别逻辑，直接用 tokenizer.json 构造一个 fast tokenizer。
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            unk_token="[UNK]",
            mask_token="[MASK]",
        )

        return tokenizer


def torch_load_checkpoint(model_path: str) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"checkpoint not found: {model_path}")
    try:
        return torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location="cpu")


# 从 checkpoint 中提取 state_dict 和模型配置；若无训练配置则回退到 fallback_config。
def get_state_dict_and_config(checkpoint, fallback_config: ModelConfig) -> Tuple[dict, ModelConfig, dict]:
    metadata = {}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = checkpoint
        train_config = checkpoint.get("training_config") or {}
        config = ModelConfig(
            d_model=int(train_config.get("d_model", fallback_config.d_model)),
            d_ff=int(train_config.get("d_ff", fallback_config.d_ff)),
            num_heads=int(train_config.get("num_heads", fallback_config.num_heads)),
            num_layers=int(train_config.get("num_layers", fallback_config.num_layers)),
            dropout=float(train_config.get("dropout", fallback_config.dropout)),
            max_src_len=int(train_config.get("max_src_len", fallback_config.max_src_len)),
            max_new_tokens=fallback_config.max_new_tokens,
        )
        print(f"loaded checkpoint epoch={checkpoint.get('epoch', 'unknown')}")
        print(f"checkpoint global_step={checkpoint.get('global_step', 'unknown')}")
        return state_dict, config, metadata

    return checkpoint, fallback_config, metadata


# 从 checkpoint 权重的形状反推训练时的词表大小，用于校验 tokenizer 是否与 checkpoint 匹配。
def infer_vocab_size_from_state_dict(state_dict: dict) -> Optional[int]:
    weight = state_dict.get("generator.proj.weight")
    if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
        return int(weight.shape[0])
    weight = state_dict.get("src_embed.0.embedding.weight")
    if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
        return int(weight.shape[0])
    return None


def load_trained_model(
    model_path: str,
    tokenizer,
    device: torch.device,
    fallback_config: ModelConfig,
) -> Tuple[EncoderDecoderTransformer, ModelConfig]:
    checkpoint = torch_load_checkpoint(model_path)
    state_dict, config, metadata = get_state_dict_and_config(checkpoint, fallback_config)

    tokenizer_vocab_size = len(tokenizer)
    checkpoint_vocab_size = infer_vocab_size_from_state_dict(state_dict)
    vocab_size = checkpoint_vocab_size or tokenizer_vocab_size
    if checkpoint_vocab_size is not None and checkpoint_vocab_size != tokenizer_vocab_size:
        raise ValueError(
            f"tokenizer vocab_size={tokenizer_vocab_size} does not match "
            f"checkpoint vocab_size={checkpoint_vocab_size}. Please use the tokenizer "
            "saved with this checkpoint."
        )

    model = make_transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        num_layers=config.num_layers,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_heads=config.num_heads,
        dropout=config.dropout,
    )

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    print(
        "model config | "
        f"d_model={config.d_model}, d_ff={config.d_ff}, heads={config.num_heads}, "
        f"layers={config.num_layers}, dropout={config.dropout}, vocab_size={vocab_size}"
    )

    return model, config


# ============================================================
# 4. Greedy Translate
# ============================================================

@torch.no_grad()
def greedy_translate(
    model: EncoderDecoderTransformer,
    tokenizer,
    text: str,
    device: torch.device,
    max_src_len: int,
    max_new_tokens: int,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    min_new_tokens: int = 1,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> str:
    model.eval()

    # 1. 源句编码：text -> src token ids
    src_content_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_src_len - 2,
    )

    src_ids = [bos_id] + src_content_ids + [eos_id]

    src = torch.tensor(
        src_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    # src: [1, S]

    src_mask = make_src_mask(src, pad_id)
    # src_mask: [1, 1, 1, S]

    # 2. Encoder 只执行一次，得到 memory
    autocast_enabled = use_amp and device.type == "cuda"
    def maybe_autocast():
        if autocast_enabled:
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return nullcontext()

    with maybe_autocast():
        memory = model.encode(src, src_mask)
    # memory: [1, S, d_model]

    # 3. Decoder 从 BOS 开始自回归生成
    ys = torch.tensor(
        [[bos_id]],
        dtype=torch.long,
        device=device,
    )
    # ys: [1, 1]

    for _ in range(max_new_tokens):
        tgt_mask = make_tgt_mask(ys, pad_id)
        # tgt_mask: [1, 1, T, T]

        with maybe_autocast():
            decoder_out = model.decode(
                tgt_in=ys,
                memory=memory,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
            )
            # decoder_out: [1, T, d_model]

            logits = model.generator(decoder_out)
        # logits: [1, T, vocab_size]

        next_token_logits = logits[:, -1, :].clone()
        # next_token_logits: [1, vocab_size]

        # 避免生成 PAD / BOS。
        next_token_logits[:, pad_id] = -1e9
        next_token_logits[:, bos_id] = -1e9

        # 前几个 token 不允许 EOS，避免空输出。
        generated_len = ys.size(1) - 1
        if generated_len < min_new_tokens:
            next_token_logits[:, eos_id] = -1e9

        next_token = torch.argmax(next_token_logits, dim=-1)
        # next_token: [1]

        next_token_id = next_token.item()

        ys = torch.cat(
            [ys, next_token.unsqueeze(1)],
            dim=1,
        )
        # ys: [1, T+1]

        if next_token_id == eos_id:
            break

    # 4. 后处理：去掉 BOS / EOS，再 decode
    generated_ids = ys[0].tolist()

    # 去掉开头 BOS
    generated_ids = generated_ids[1:]

    # 如果有 EOS，则截断到 EOS 之前
    if eos_id in generated_ids:
        generated_ids = generated_ids[:generated_ids.index(eos_id)]

    translated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return translated_text


# ============================================================
# 5. Main Inference Loop
# ============================================================

def main() -> None:
    global PAD_ID, BOS_ID, EOS_ID

    model_dir = MODEL_DIR
    model_path = MODEL_PATH
    tokenizer_dir = TOKENIZER_DIR

    if not os.path.isdir(tokenizer_dir) and os.path.isdir(FALLBACK_TOKENIZER_DIR):
        print(f"tokenizer dir not found under model dir, fallback to: {FALLBACK_TOKENIZER_DIR}")
        tokenizer_dir = FALLBACK_TOKENIZER_DIR

    device = get_best_device()

    print(f"device={device}")
    print(f"model_dir={model_dir}")
    print(f"checkpoint={model_path}")
    print(f"tokenizer_dir={tokenizer_dir}")

    if device.type == "cuda":
        print(f"Using CUDA acceleration: {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        print("Using Apple Silicon MPS acceleration for inference.")
    else:
        print("Using CPU for inference.")

    tokenizer = load_local_tokenizer(tokenizer_dir)

    PAD_ID = tokenizer.pad_token_id
    BOS_ID = tokenizer.cls_token_id
    EOS_ID = tokenizer.sep_token_id

    assert PAD_ID is not None, "PAD_ID is None. Please check tokenizer."
    assert BOS_ID is not None, "BOS_ID is None. Please check tokenizer."
    assert EOS_ID is not None, "EOS_ID is None. Please check tokenizer."

    print(f"vocab_size={len(tokenizer)}")
    print(f"PAD_ID={PAD_ID}, BOS_ID={BOS_ID}, EOS_ID={EOS_ID}")

    fallback_config = ModelConfig(
        max_src_len=MAX_SRC_LEN,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    model, model_config = load_trained_model(
        model_path=model_path,
        tokenizer=tokenizer,
        device=device,
        fallback_config=fallback_config,
    )
    model_config.max_src_len = MAX_SRC_LEN
    model_config.max_new_tokens = MAX_NEW_TOKENS

    print("model loaded successfully.")

    # 先跑几个固定例子，方便确认推理闭环是否正常。
    example_texts = [
        "I like machine learning.",
        "This is a small Transformer model.",
        "The tokenizer split the word into many subwords.",
        "The model that we trained yesterday can translate technical terms.",
        "The sentence that contains several clauses is hard to translate.",
        "If I had known the training would take so long, I would have started earlier.",
        "Had I known the model was unstable, I would have used a smaller learning rate.",
        "The more data the model sees, the better it usually becomes.",
        "Even though the sentence is long, the translation should be natural.",
        "The decoder repeated the same phrase several times."
    ]

    use_amp = device.type == "cuda"

    def translate_and_print(text: str, prefix: str = "EN") -> None:
        translation = greedy_translate(
            model=model,
            tokenizer=tokenizer,
            text=text,
            device=device,
            max_src_len=model_config.max_src_len,
            max_new_tokens=model_config.max_new_tokens,
            pad_id=PAD_ID,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            min_new_tokens=MIN_NEW_TOKENS,
            use_amp=use_amp,
        )
        print(f"{prefix}: {text}")
        print(f"ZH: {translation}")
        print("-" * 60)

    print("\n===== Fixed Examples =====")
    for text in example_texts:
        translate_and_print(text)

    # 交互式推理
    print("\n===== Interactive Translation =====")
    print("Input English text. Type 'q' or 'quit' to exit.")

    while True:
        try:
            text = input("\nEN> ").strip()
        except EOFError:
            print("\nstdin closed, bye.")
            break

        if text.lower() in {"q", "quit", "exit"}:
            print("bye.")
            break

        if not text:
            continue

        translation = greedy_translate(
            model=model,
            tokenizer=tokenizer,
            text=text,
            device=device,
            max_src_len=model_config.max_src_len,
            max_new_tokens=model_config.max_new_tokens,
            pad_id=PAD_ID,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            min_new_tokens=MIN_NEW_TOKENS,
            use_amp=use_amp,
        )

        print(f"ZH> {translation}")


if __name__ == "__main__":
    main()
