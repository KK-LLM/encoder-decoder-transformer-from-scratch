import argparse
import os
import math
from contextlib import nullcontext
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizerFast


# ============================================================
# 0. 路径与模型配置
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V1_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(V1_ROOT, "checkpoints", "checkpoint_best.pt")
TOKENIZER_DIR = os.path.join(V1_ROOT, "tokenizer", "en_zh_stage_tokenizer_48000")

# 推理解码长度配置。源句长度从 checkpoint 的 training_config 读取。
MAX_NEW_TOKENS = 96
MIN_NEW_TOKENS = 1
DEFAULT_BATCH_SIZE = 16


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


# 从 checkpoint 中提取 state_dict 和模型配置；V1 推理要求 checkpoint 内含 training_config。
def get_state_dict_and_config(checkpoint) -> Tuple[dict, dict]:
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint must be a training checkpoint with model_state_dict.")

    train_config = checkpoint.get("training_config")
    if not isinstance(train_config, dict):
        raise ValueError("checkpoint missing training_config; cannot infer model config safely.")

    required_keys = (
        "d_model",
        "d_ff",
        "num_heads",
        "num_layers",
        "dropout",
        "max_src_len",
    )
    missing_keys = [key for key in required_keys if key not in train_config]
    if missing_keys:
        raise ValueError(
            "checkpoint training_config missing required keys: "
            + ", ".join(missing_keys)
        )

    config = {
        "d_model": int(train_config["d_model"]),
        "d_ff": int(train_config["d_ff"]),
        "num_heads": int(train_config["num_heads"]),
        "num_layers": int(train_config["num_layers"]),
        "dropout": float(train_config["dropout"]),
        "max_src_len": int(train_config["max_src_len"]),
    }

    print(f"loaded checkpoint epoch={checkpoint.get('epoch', 'unknown')}")
    print(f"checkpoint global_step={checkpoint.get('global_step', 'unknown')}")

    return checkpoint["model_state_dict"], config


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
) -> Tuple[EncoderDecoderTransformer, dict]:
    checkpoint = torch_load_checkpoint(model_path)
    state_dict, config = get_state_dict_and_config(checkpoint)

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
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
    )

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    print(
        "model config | "
        f"d_model={config['d_model']}, d_ff={config['d_ff']}, heads={config['num_heads']}, "
        f"layers={config['num_layers']}, dropout={config['dropout']}, vocab_size={vocab_size}, "
        f"max_src_len={config['max_src_len']}"
    )

    return model, config


# ============================================================
# 4. Batch Greedy Translate
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Encoder-Decoder English-to-Chinese inference."
    )
    parser.add_argument(
        "--checkpoint",
        default=MODEL_PATH,
        help="Path to the V1 checkpoint.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=TOKENIZER_DIR,
        help="Path to the tokenizer directory used by the V1 checkpoint.",
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Text file with one English sentence per non-empty line.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size used for file inference.",
    )
    return parser.parse_args()


@torch.inference_mode()
def batch_greedy_translate(
    model: EncoderDecoderTransformer,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_src_len: int,
    max_new_tokens: int,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    min_new_tokens: int = 1,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> List[str]:
    if not texts:
        return []

    model.eval()

    src_tensors = []
    for text in texts:
        src_content_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_src_len - 2,
        )
        src_ids = [bos_id] + src_content_ids + [eos_id]
        src_tensors.append(torch.tensor(src_ids, dtype=torch.long))

    batch_size = len(src_tensors)
    src_seq_len = max(src_tensor.numel() for src_tensor in src_tensors)
    src = torch.full(
        (batch_size, src_seq_len),
        fill_value=pad_id,
        dtype=torch.long,
        device=device,
    )
    for index, src_tensor in enumerate(src_tensors):
        src[index, :src_tensor.numel()] = src_tensor.to(device)

    src_mask = make_src_mask(src, pad_id)

    autocast_enabled = use_amp and device.type == "cuda"

    def maybe_autocast():
        if autocast_enabled:
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        return nullcontext()

    with maybe_autocast():
        memory = model.encode(src, src_mask)

    ys = torch.full(
        (batch_size, 1),
        fill_value=bos_id,
        dtype=torch.long,
        device=device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    eos_tokens = torch.full(
        (batch_size,),
        fill_value=eos_id,
        dtype=torch.long,
        device=device,
    )

    for _ in range(max_new_tokens):
        tgt_mask = make_tgt_mask(ys, pad_id)

        with maybe_autocast():
            decoder_out = model.decode(
                tgt_in=ys,
                memory=memory,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
            )
            logits = model.generator(decoder_out)

        next_token_logits = logits[:, -1, :].float().clone()
        next_token_logits[:, pad_id] = -1e9
        next_token_logits[:, bos_id] = -1e9

        generated_len = ys.size(1) - 1
        if generated_len < min_new_tokens:
            next_token_logits[:, eos_id] = -1e9

        next_token = torch.argmax(next_token_logits, dim=-1)
        next_token = torch.where(finished, eos_tokens, next_token)

        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == eos_id)

        if finished.all().item():
            break

    translations = []
    for generated_ids in ys.tolist():
        generated_ids = generated_ids[1:]
        if eos_id in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(eos_id)]

        translated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        translations.append(translated_text)

    return translations


def load_input_texts(input_file: str) -> List[str]:
    with open(input_file, "r", encoding="utf-8") as file:
        texts = [line.strip() for line in file if line.strip()]

    if not texts:
        raise ValueError(f"input file has no non-empty English lines: {input_file}")

    return texts


def run_file_batch_translation(
    input_file: str,
    batch_size: int,
    model: EncoderDecoderTransformer,
    tokenizer,
    device: torch.device,
    max_src_len: int,
    max_new_tokens: int,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    min_new_tokens: int,
    use_amp: bool,
) -> None:
    texts = load_input_texts(input_file)

    print("\n===== File Batch Translation =====")
    print(f"input_file={input_file}")
    print(f"sentences={len(texts)}, batch_size={batch_size}")

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        translations = batch_greedy_translate(
            model=model,
            tokenizer=tokenizer,
            texts=batch_texts,
            device=device,
            max_src_len=max_src_len,
            max_new_tokens=max_new_tokens,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            min_new_tokens=min_new_tokens,
            use_amp=use_amp,
        )

        for offset, (text, translation) in enumerate(
            zip(batch_texts, translations),
            start=start + 1,
        ):
            print(f"[{offset}] EN: {text}")
            print(f"[{offset}] ZH: {translation}")
            print("-" * 60)


# ============================================================
# 5. Main Inference Loop
# ============================================================

def main() -> None:
    global PAD_ID, BOS_ID, EOS_ID

    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    model_path = args.checkpoint
    tokenizer_dir = args.tokenizer_dir
    model_dir = os.path.dirname(model_path) or "."

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

    model, model_config = load_trained_model(
        model_path=model_path,
        tokenizer=tokenizer,
        device=device,
    )

    print("model loaded successfully.")
    use_amp = device.type == "cuda"

    run_file_batch_translation(
        input_file=args.input_file,
        batch_size=args.batch_size,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_src_len=model_config["max_src_len"],
        max_new_tokens=MAX_NEW_TOKENS,
        pad_id=PAD_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        min_new_tokens=MIN_NEW_TOKENS,
        use_amp=use_amp,
    )


if __name__ == "__main__":
    main()
