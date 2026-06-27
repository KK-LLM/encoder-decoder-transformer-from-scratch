import argparse
import os
import math
import random
import shutil
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer

# 初始占位，后续会根据 tokenizer 自动更新。
PAD_ID = 0
BOS_ID = 101
EOS_ID = 102


def make_src_mask(src: torch.Tensor, pad_id: int) -> torch.Tensor:
    # src: [B, S]
    # src_mask: [B, 1, 1, S]
    # True means this source token can be attended to.
    src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2)
    return src_mask


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # causal_mask: [1, 1, T, T]
    # True means the target position can attend to that key position.
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


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        layer_norm_out = self.gamma * x_hat + self.beta

        return layer_norm_out


# output = LayerNorm(residual + Dropout(sublayer_out))
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

        # pe: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 1 / 10000^(2i / d_model) = 10000^(-2i / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        # position: [max_len, 1]
        # div_term: [d_model / 2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe: [max_len, d_model] -> [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # pe 是固定位置编码，不参与训练更新，因此注册为 buffer。
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


# dropout(softmax(qk^T/√d_head))v
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
        # mask: True means allowed; False means masked.
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
    """
    Original Transformer Post-LN Encoder Layer:

        self_attn_out = SelfAttention(x, x, x)
        x = LayerNorm(x + Dropout(self_attn_out))

        ffn_out = FFN(x)
        x = LayerNorm(x + Dropout(ffn_out))
    """

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
    """
    Original Transformer Post-LN Decoder Layer:

        self_attn_out = MaskedSelfAttention(x, x, x)
        x = LayerNorm(x + Dropout(self_attn_out))

        cross_attn_out = CrossAttention(x, memory, memory)
        x = LayerNorm(x + Dropout(cross_attn_out))

        ffn_out = FFN(x)
        x = LayerNorm(x + Dropout(ffn_out))
    """

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


# decoder_out -> Linear -> logits
# loss = CrossEntropyLoss(logits, target)
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
        num_layers: int = 2,
        d_model: int = 64,
        d_ff: int = 128,
        num_heads: int = 4,
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

    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return model


# 预先将本地 JSONL translation 样本转换成 src/tgt token ids。
class TranslationDataset(Dataset):
    def __init__(
            self,
            hf_dataset,
            tokenizer,
            source_lang: str = "en",
            target_lang: str = "zh",
            max_src_len: int = 64,
            max_tgt_len: int = 64,
            bos_id: int = BOS_ID,
            eos_id: int = EOS_ID,
            print_every: int = 10000,
    ):
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.tokenizer = tokenizer
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.bos_id = bos_id
        self.eos_id = eos_id

        total = len(hf_dataset)
        for index, item in enumerate(hf_dataset):
            src_text, tgt_text = self.extract_source_target(item)

            if not isinstance(src_text, str) or not src_text.strip():
                continue

            if not isinstance(tgt_text, str) or not tgt_text.strip():
                continue

            src = self.encode_text(src_text, self.max_src_len)
            tgt = self.encode_text(tgt_text, self.max_tgt_len)

            # 至少要包含 BOS + 正文 + EOS
            if src.numel() < 3 or tgt.numel() < 3:
                continue

            self.samples.append((src, tgt))

            if print_every > 0 and (index + 1) % print_every == 0:
                print(
                    f"tokenized {index + 1}/{total} raw samples | "
                    f"kept={len(self.samples)}"
                )

        print(f"finished tokenization | raw={total} | kept={len(self.samples)}")

    def extract_source_target(self, item) -> Tuple[Optional[str], Optional[str]]:
        translation = item.get("translation") if isinstance(item, dict) else None

        if isinstance(translation, dict):
            return (
                translation.get(self.source_lang),
                translation.get(self.target_lang),
            )

        return item.get("source"), item.get("target")

    def __len__(self) -> int:
        return len(self.samples)

    def encode_text(self, text: str, max_len: int) -> torch.Tensor:
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len - 2,
        )

        token_ids = [self.bos_id] + token_ids + [self.eos_id]

        return torch.tensor(token_ids, dtype=torch.long)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[index]


def collate_translation_batch(
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    src_list, tgt_list = zip(*batch)

    src = pad_sequence(
        src_list,
        batch_first=True,
        padding_value=PAD_ID,
    )

    tgt = pad_sequence(
        tgt_list,
        batch_first=True,
        padding_value=PAD_ID,
    )

    return src, tgt


# learning_rate = factor × d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))
def noam_rate(
        step: int,  # 当前训练步数；通常每执行一次 optimizer.step() / scheduler.step()，step 增加 1
        d_model: int,  # Transformer 的 hidden size / 模型维度，用于按 d_model^(-0.5) 缩放学习率
        warmup_steps: int,  # warmup 阶段的步数；在此之前学习率逐步升高，之后开始衰减
        factor: float = 1.0  # 学习率整体缩放系数；用于整体放大或缩小 Noam 学习率
) -> float:
    step = max(step, 1)

    learning_rate = factor * (d_model ** -0.5) * min(
        step ** -0.5,
        step * (warmup_steps ** -1.5),
    )

    return learning_rate


def compute_loss(
        logits: torch.Tensor,
        tgt_out: torch.Tensor,
        pad_id: int,
        label_smoothing: float = 0.1,
) -> torch.Tensor:
    # logits:  [B, T, vocab_size]
    # tgt_out: [B, T]
    vocab_size = logits.size(-1)

    logits_flat = logits.reshape(-1, vocab_size)
    tgt_out_flat = tgt_out.reshape(-1)

    loss = F.cross_entropy(
        logits_flat,
        tgt_out_flat,
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )

    return loss


def train_one_epoch(
        model: EncoderDecoderTransformer,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        device: torch.device,
        pad_id: int,
        label_smoothing: float,
        use_amp: bool,
        amp_dtype: torch.dtype,
) -> float:
    model.train()
    total_loss = 0.0

    for src, tgt in dataloader:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_src_mask(src, pad_id)
        tgt_mask = make_tgt_mask(tgt_in, pad_id)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=use_amp,
        ):
            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = compute_loss(
                logits=logits,
                tgt_out=tgt_out,
                pad_id=pad_id,
                label_smoothing=label_smoothing,
            )

        loss.backward()

        # 防止真实数据上偶发梯度过大，便于最小脚本稳定训练。
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def evaluate(
        model: EncoderDecoderTransformer,
        dataloader: DataLoader,
        device: torch.device,
        pad_id: int,
        label_smoothing: float,
        use_amp: bool,
        amp_dtype: torch.dtype,
) -> float:
    model.eval()
    total_loss = 0.0

    for src, tgt in dataloader:
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        src_mask = make_src_mask(src, pad_id)
        tgt_mask = make_tgt_mask(tgt_in, pad_id)

        with torch.amp.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=use_amp,
        ):
            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss = compute_loss(
                logits=logits,
                tgt_out=tgt_out,
                pad_id=pad_id,
                label_smoothing=label_smoothing,
            )

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


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
        min_new_tokens: int = 3,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.bfloat16,
) -> str:
    model.eval()

    src_content_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_src_len - 2,
    )

    src_ids = [bos_id] + src_content_ids + [eos_id]
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    # src: [1, S]

    src_mask = make_src_mask(src, pad_id)
    # src_mask: [1, 1, 1, S]

    with torch.amp.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=use_amp,
    ):
        memory = model.encode(src, src_mask)
        # memory: [1, S, d_model]

    ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    # ys: [1, 1]

    for _ in range(max_new_tokens):
        tgt_mask = make_tgt_mask(ys, pad_id)
        # tgt_mask: [1, 1, T, T]

        with torch.amp.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=use_amp,
        ):
            decoder_out = model.decode(
                tgt_in=ys,
                memory=memory,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
            )
            # decoder_out: [1, T, d_model]

            logits = model.generator(decoder_out)
            # logits: [1, T, vocab_size]

        next_token_logits = logits[:, -1, :].float()
        # next_token_logits: [1, vocab_size]

        # 避免生成 PAD / BOS。
        next_token_logits[:, pad_id] = -1e9
        next_token_logits[:, bos_id] = -1e9

        # 早期不允许直接生成 EOS，避免 ZH 为空。
        generated_len = ys.size(1) - 1
        if generated_len < min_new_tokens:
            next_token_logits[:, eos_id] = -1e9

        next_token = torch.argmax(next_token_logits, dim=-1)
        # next_token: [1]

        next_token_id = next_token.item()

        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        # ys: [1, T+1]

        if next_token_id == eos_id:
            break

    generated_ids = ys[0].tolist()

    # 去掉开头 BOS；如果有 EOS，也截断到 EOS 之前。
    generated_ids = generated_ids[1:]
    if eos_id in generated_ids:
        generated_ids = generated_ids[:generated_ids.index(eos_id)]

    translated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return translated_text


def move_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {key: move_to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [move_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_cpu(value) for value in obj)
    return obj


def optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def get_rng_state() -> dict:
    rng_state = {
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        rng_state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    return move_to_cpu(rng_state)


def restore_rng_state(rng_state: Optional[dict]) -> None:
    if not rng_state:
        return

    python_random_state = rng_state.get("python_random_state")
    if python_random_state is not None:
        random.setstate(python_random_state)

    torch_rng_state = rng_state.get("torch_rng_state")
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state)

    cuda_rng_state_all = rng_state.get("cuda_rng_state_all")
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng_state_all)


def atomic_torch_save(obj: dict, path: str) -> None:
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def torch_load_checkpoint(checkpoint_path: str) -> dict:
    try:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def warn_if_config_mismatch(checkpoint: dict, current_config: dict) -> None:
    old_config = checkpoint.get("training_config")
    if not isinstance(old_config, dict):
        return

    critical_keys = [
        "vocab_size",
        "d_model",
        "d_ff",
        "num_heads",
        "num_layers",
        "max_src_len",
        "max_tgt_len",
    ]
    for key in critical_keys:
        if key in old_config and key in current_config and old_config[key] != current_config[key]:
            print(
                f"[checkpoint warning] config mismatch for {key}: "
                f"checkpoint={old_config[key]} current={current_config[key]}"
            )


def save_training_checkpoint(
        model: EncoderDecoderTransformer,
        tokenizer,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        epoch: int,
        output_dir: str,
        training_config: dict,
        train_loss_history: List[float],
        valid_loss_history: List[float],
        global_step: int,
        checkpoint_name: Optional[str] = None,
        is_final: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_dir)

    checkpoint = {
        "checkpoint_version": 2,
        "epoch": epoch,
        "global_step": global_step,
        "is_final": is_final,
        "model_state_dict": move_to_cpu(model.state_dict()),
        "optimizer_state_dict": move_to_cpu(optimizer.state_dict()),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": get_rng_state(),
        "training_config": dict(training_config),
        "train_loss_history": list(train_loss_history),
        "valid_loss_history": list(valid_loss_history),
        "tokenizer_dir": tokenizer_dir,
        "pad_id": PAD_ID,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
    }

    latest_path = os.path.join(output_dir, "checkpoint_latest.pt")
    atomic_torch_save(checkpoint, latest_path)
    print(f"saved latest checkpoint to {latest_path}")

    if checkpoint_name is not None:
        checkpoint_path = os.path.join(output_dir, checkpoint_name)
        atomic_torch_save(checkpoint, checkpoint_path)
        print(f"saved checkpoint to {checkpoint_path}")


def load_training_checkpoint(
        checkpoint_path: str,
        model: EncoderDecoderTransformer,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        device: torch.device,
        training_config: dict,
) -> Tuple[int, int, List[float], List[float]]:
    checkpoint = torch_load_checkpoint(checkpoint_path)
    warn_if_config_mismatch(checkpoint, training_config)

    model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_state_to_device(optimizer, device)
    else:
        print("[checkpoint warning] optimizer_state_dict not found; optimizer starts fresh.")

    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        print("[checkpoint warning] scheduler_state_dict not found; scheduler starts fresh.")

    restore_rng_state(checkpoint.get("rng_state"))

    last_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", scheduler.last_epoch))
    train_loss_history = list(checkpoint.get("train_loss_history", []))
    valid_loss_history = list(checkpoint.get("valid_loss_history", []))

    print(
        f"loaded checkpoint from {checkpoint_path} | "
        f"last_epoch={last_epoch} | global_step={global_step}"
    )

    return last_epoch, global_step, train_loss_history, valid_loss_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the v1 curriculum Encoder-Decoder Transformer."
    )
    parser.add_argument(
        "--curriculum-root",
        default=os.path.join("encoder_decoder", "v1_curriculum", "data"),
        help="Directory containing stage1..stage5 train.jsonl and final_eval.jsonl.",
    )
    parser.add_argument(
        "--stage",
        default="stage1",
        choices=["stage1", "stage2", "stage3", "stage4", "stage5"],
        help="Curriculum stage to train.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=os.path.join(
            "encoder_decoder",
            "v1_curriculum",
            "tokenizer",
            "en_zh_stage_tokenizer_48000",
        ),
        help="Prebuilt 48k tokenizer directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Training output directory. Defaults to outputs/v1_curriculum/<stage>.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Optional checkpoint path for continuing training.",
    )
    parser.add_argument(
        "--stage-train-epochs",
        type=int,
        default=20,
        help="Number of epochs to run for this stage invocation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=168,
        help="Training and validation batch size.",
    )
    return parser.parse_args()


def load_prebuilt_tokenizer(tokenizer_output_dir: str):
    if not os.path.isdir(tokenizer_output_dir):
        raise FileNotFoundError(
            "curriculum training must use the prebuilt local 48000 tokenizer; "
            f"missing tokenizer_output_dir={tokenizer_output_dir}"
        )
    print(f"loading tokenizer from local dir: {tokenizer_output_dir}")
    return AutoTokenizer.from_pretrained(tokenizer_output_dir, use_fast=True)


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")
    device = torch.device("cuda")

    # ============================================================
    # 1. 基础配置
    # ============================================================
    curriculum_root = args.curriculum_root
    curriculum_stage = args.stage
    assert curriculum_stage in {"stage1", "stage2", "stage3", "stage4", "stage5"}

    tokenizer_output_dir = args.tokenizer_dir

    source_lang = "en"
    target_lang = "zh"

    train_jsonl_path = os.path.join(curriculum_root, curriculum_stage, "train.jsonl")
    valid_jsonl_path = os.path.join(curriculum_root, "final_eval.jsonl")
    detailed_valid_jsonl_paths = {
        "valid_easy": os.path.join(curriculum_root, curriculum_stage, "valid_easy.jsonl"),
        "valid_tech": os.path.join(curriculum_root, curriculum_stage, "valid_tech.jsonl"),
        "valid_logic": os.path.join(curriculum_root, curriculum_stage, "valid_logic.jsonl"),
    }
    detailed_valid_eval_every_epochs = 2

    max_src_len = 96
    max_tgt_len = 96

    d_model = 768
    d_ff = 3072
    num_heads = 12
    num_layers = 10
    dropout = 0.08

    batch_size = args.batch_size

    stage_train_epochs = args.stage_train_epochs
    num_epochs = stage_train_epochs
    warmup_steps = 7000
    label_smoothing = 0.03

    max_train_samples = None
    max_valid_samples = None

    # CUDA/A800 使用 BF16 autocast。
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16

    # DataLoader 配置：只做基础 GPU 数据加载优化。
    num_workers = min(8, os.cpu_count() or 1)
    pin_memory = device.type == "cuda"

    output_dir = args.output_dir or os.path.join("outputs", "v1_curriculum", curriculum_stage)
    # None 表示从零开始训练；否则必须显式给出存在的 checkpoint 路径。
    resume_checkpoint_path = args.resume_checkpoint
    # resume_checkpoint_path = "/root/autodl-tmp/minimal_transformer_en_zh_curriculum_v1_stage4_outputs_new/checkpoint_epoch_48.pt"

    if resume_checkpoint_path is None:
        resolved_resume_checkpoint_path = None
    elif os.path.isfile(resume_checkpoint_path):
        resolved_resume_checkpoint_path = resume_checkpoint_path
    else:
        raise FileNotFoundError(f"resume_checkpoint_path not found: {resume_checkpoint_path}")

    print(f"device={device}")
    if resolved_resume_checkpoint_path is not None:
        print(f"resume checkpoint={resolved_resume_checkpoint_path}")
    else:
        print("resume checkpoint=None")

    if device.type == "cuda":
        print(f"Using CUDA acceleration: {torch.cuda.get_device_name(0)}")
        print(f"AMP enabled={use_amp}, dtype={amp_dtype}")
    else:
        print("CUDA is not available. Training will be much slower.")

    # ============================================================
    # 2. 加载本地 JSONL 中英文翻译数据集
    # ============================================================
    data_files = {
        "train": train_jsonl_path,
        "validation": valid_jsonl_path,
    }
    for split_name, jsonl_path in detailed_valid_jsonl_paths.items():
        if os.path.exists(jsonl_path):
            data_files[split_name] = jsonl_path
        else:
            print(f"[detailed validation] missing {split_name}: {jsonl_path}")

    raw_dataset = load_dataset(
        "json",
        data_files=data_files,
    )

    train_raw = raw_dataset["train"]
    valid_raw = raw_dataset["validation"]
    detailed_valid_raw = {
        split_name: raw_dataset[split_name]
        for split_name in detailed_valid_jsonl_paths
        if split_name in raw_dataset
    }

    if max_train_samples is not None:
        train_raw = train_raw.select(range(min(max_train_samples, len(train_raw))))

    if max_valid_samples is not None:
        valid_raw = valid_raw.select(range(min(max_valid_samples, len(valid_raw))))
        detailed_valid_raw = {
            split_name: split_raw.select(range(min(max_valid_samples, len(split_raw))))
            for split_name, split_raw in detailed_valid_raw.items()
        }

    print(f"train raw samples={len(train_raw)}")
    print(f"valid raw samples={len(valid_raw)}")
    for split_name, split_raw in detailed_valid_raw.items():
        print(f"{split_name} raw samples={len(split_raw)}")

    # ============================================================
    # 3. 加载本地 tokenizer
    # ============================================================
    tokenizer = load_prebuilt_tokenizer(tokenizer_output_dir=tokenizer_output_dir)

    global PAD_ID, BOS_ID, EOS_ID
    PAD_ID = tokenizer.pad_token_id
    BOS_ID = tokenizer.cls_token_id
    EOS_ID = tokenizer.sep_token_id

    assert PAD_ID is not None, "PAD_ID is None. Please check tokenizer.pad_token."
    assert BOS_ID is not None, "BOS_ID is None. Please check tokenizer.cls_token."
    assert EOS_ID is not None, "EOS_ID is None. Please check tokenizer.sep_token."

    # 用 len(tokenizer) 更稳妥，包含完整 tokenizer 词表规模。
    vocab_size = len(tokenizer)
    expected_ids = {
        "vocab_size": (vocab_size, 48000),
        "pad_id": (PAD_ID, 0), "unk_id": (tokenizer.unk_token_id, 1),
        "bos_id": (BOS_ID, 2), "eos_id": (EOS_ID, 3),
        "mask_id": (tokenizer.mask_token_id, 4),
    }
    for name, (actual, expected) in expected_ids.items():
        assert actual == expected, f"{name} mismatch: actual={actual}, expected={expected}"

    training_config = {
        "tokenizer_output_dir": tokenizer_output_dir,
        "curriculum_root": curriculum_root,
        "curriculum_stage": curriculum_stage,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "train_jsonl_path": train_jsonl_path,
        "valid_jsonl_path": valid_jsonl_path,
        "max_src_len": max_src_len,
        "max_tgt_len": max_tgt_len,
        "d_model": d_model,
        "d_ff": d_ff,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
        "batch_size": batch_size,
        "stage_train_epochs": stage_train_epochs,
        "num_epochs": num_epochs,
        "warmup_steps": warmup_steps,
        "label_smoothing": label_smoothing,
        "vocab_size": vocab_size,
        "pad_id": PAD_ID,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
    }

    print(f"tokenizer_output_dir={tokenizer_output_dir}")
    print(f"new tokenizer vocab_size={vocab_size}")
    print(f"PAD_ID={PAD_ID}, BOS_ID={BOS_ID}, EOS_ID={EOS_ID}")

    # ============================================================
    # 4. 构造 Dataset / DataLoader
    # ============================================================
    train_dataset = TranslationDataset(
        hf_dataset=train_raw,
        tokenizer=tokenizer,
        source_lang=source_lang,
        target_lang=target_lang,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
    )

    valid_dataset = TranslationDataset(
        hf_dataset=valid_raw,
        tokenizer=tokenizer,
        source_lang=source_lang,
        target_lang=target_lang,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        print_every=0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_translation_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_translation_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    detailed_valid_loaders = {}
    for split_name, split_raw in detailed_valid_raw.items():
        split_dataset = TranslationDataset(
            hf_dataset=split_raw,
            tokenizer=tokenizer,
            source_lang=source_lang,
            target_lang=target_lang,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            print_every=0,
        )
        detailed_valid_loaders[split_name] = DataLoader(
            split_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_translation_batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    print(f"train dataset size={len(train_dataset)}")
    print(f"valid dataset size={len(valid_dataset)}")
    for split_name, split_loader in detailed_valid_loaders.items():
        print(f"{split_name} dataset size={len(split_loader.dataset)}")
    print(f"steps per epoch={len(train_loader)}")

    # ============================================================
    # 5. 创建 Encoder-Decoder Transformer
    # ============================================================
    model = make_transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        num_layers=num_layers,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)

    print(
        f"model config | "
        f"d_model={d_model}, d_ff={d_ff}, heads={num_heads}, "
        f"layers={num_layers}, dropout={dropout}, vocab_size={vocab_size}, "
        f"batch_size={batch_size}, max_len=({max_src_len}, {max_tgt_len})"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    def noam_lr_lambda(current_step: int) -> float:
        lr_multiplier = noam_rate(
            step=current_step,
            d_model=d_model,
            warmup_steps=warmup_steps,
        )
        return lr_multiplier

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=noam_lr_lambda,
    )

    start_epoch = 1
    global_step = 0
    train_loss_history: List[float] = []
    valid_loss_history: List[float] = []

    if resolved_resume_checkpoint_path is not None:
        (
            last_epoch,
            global_step,
            train_loss_history,
            valid_loss_history,
        ) = load_training_checkpoint(
            checkpoint_path=resolved_resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            training_config=training_config,
        )
        start_epoch = last_epoch + 1
        num_epochs = last_epoch + stage_train_epochs
        training_config["num_epochs"] = num_epochs
        print(
            f"stage_train_epochs={stage_train_epochs} | "
            f"resume_from_epoch={last_epoch} | stop_epoch={num_epochs}"
        )

    if start_epoch > num_epochs:
        print(
            f"checkpoint already reached epoch {start_epoch - 1}; "
            f"num_epochs={num_epochs}, nothing to train."
        )
        return

    # ============================================================
    # 6. 训练与验证
    # ============================================================
    best_valid_loss = float("inf")

    example_groups = {
        "basic": [
            "Please open the window in the room before work.",
            "the girl opened the window yesterday.",
            "my friend didn't open the door this morning.",
            "she should read the book.",
            "my brother put the laptop on the desk after dinner.",
            "my sister will drink a cup of coffee this morning.",
            "your friend will talk to her sister today.",
            "The meeting starts at nine today.",
        ],
        "general_logic": [
            "I will call you when the meeting is over.",
            "If you need more time, I can help you tomorrow.",
            "Because this work is important, we should do it today.",
            "The problem is simple, but the answer is not clear.",
            "The time is short, so we should start now.",
            "While you read the book, I will close the door.",
            "Although the work was difficult, my friend did it today.",
            "My sister was late, but she still did the work.",
        ],
        "technical_terms": [
            "Machine learning models need clean training data to generalize well.",
            "Deep learning models can overfit when the dataset is too small.",
            "The tokenizer splits English and Chinese sentences into stable tokens.",
            "A poor tokenizer may split rare words into many small pieces.",
            "A Transformer encoder uses self-attention to combine input tokens.",
            "The encoder reads the source sentence and produces contextual representations.",
            "The decoder uses the encoder output to generate the target sentence.",
            "Embedding layers map tokens to dense vectors before the encoder reads them.",
            "After each epoch, the training script saves a checkpoint.",
            "If the wrong checkpoint is loaded, the output may become unstable.",
            "If the learning rate is too high, the validation loss may become unstable.",
            "When validation loss decreases, the checkpoint is usually better."
        ],
        "complex_logic": [
            "Although the validation loss increased, the earlier checkpoint still produced stable translations.",
            "If the tokenizer splits rare words poorly, the decoder may lose part of the meaning.",
            "When the encoder reads a long sentence, the decoder should keep the main idea.",
            "Even though the dataset is small, clean replay examples can still help the model.",
            "The model trained on clean data handled the sentence better than the model trained on noisy data.",
            "The more repeated tokens appear in the output, the less stable the translation becomes.",
            "If the main clause is translated correctly, the sentence can still be wrong when the subordinate clause is missing.",
            "Not only should the tokenizer preserve technical terms, but the decoder should also generate natural Chinese."
        ],

    "regression_antipollution": [
            "Although the checkpoint was saved yesterday, the model still confused the notebook with the laptop.",
            "When book appears near a Transformer sentence, it should still mean an ordinary book.",
            "If the paper is on the table, it should not be confused with an academic paper.",
            "The sentence contains the phrase do not translate, but it still needs a natural Chinese translation.",
            "In this report, Transformer stays in English, while tokenizer is translated as 分词器.",
            "Even though the sentence contains checkpoint and token, door and window are still ordinary objects.",
            "If the output repeats the same token, the translation should be considered unstable.",
            "The sentence mentions next week, encoder, decoder, and a laptop, but it is still one normal sentence."
        ]
    }

    for epoch in range(start_epoch, num_epochs + 1):
        if epoch > start_epoch:
            print("\n" * 4, end="")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            pad_id=PAD_ID,
            label_smoothing=label_smoothing,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        valid_loss = evaluate(
            model=model,
            dataloader=valid_loader,
            device=device,
            pad_id=PAD_ID,
            label_smoothing=label_smoothing,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        global_step += len(train_loader)
        train_loss_history.append(float(train_loss))
        valid_loss_history.append(float(valid_loss))

        is_best = valid_loss < best_valid_loss
        if is_best:
            best_valid_loss = float(valid_loss)

        print(
            f"epoch={epoch} | "
            f"train_loss={train_loss:.4f} | "
            f"valid_loss={valid_loss:.4f} | "
            f"best_valid_loss={best_valid_loss:.4f} | "
            f"lr={current_lr:.8f}"
        )

        if detailed_valid_loaders and epoch % detailed_valid_eval_every_epochs == 0:
            detailed_parts = []
            for split_name, split_loader in detailed_valid_loaders.items():
                split_loss = evaluate(
                    model=model,
                    dataloader=split_loader,
                    device=device,
                    pad_id=PAD_ID,
                    label_smoothing=label_smoothing,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )
                detailed_parts.append(f"{split_name}_loss={split_loss:.4f}")
            print(f"epoch={epoch} | detailed_valid | " + " | ".join(detailed_parts))

        print("========== Example Translations ==========")
        for group_name, texts in example_groups.items():
            print(f"\n[{group_name}]")
            for example_text in texts:
                translation = greedy_translate(
                    model=model,
                    tokenizer=tokenizer,
                    text=example_text,
                    device=device,
                    max_src_len=max_src_len,
                    max_new_tokens=50,
                    pad_id=PAD_ID,
                    bos_id=BOS_ID,
                    eos_id=EOS_ID,
                    min_new_tokens=3,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )

                print(f"EN: {example_text}")
                print(f"ZH: {translation}")

        # 每个 epoch 都保存 latest，便于中断后继续；每 4 个 epoch 额外保存归档。
        archive_checkpoint_name = None
        if epoch % 4 == 0:
            archive_checkpoint_name = f"checkpoint_epoch_{epoch}.pt"

        save_training_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            output_dir=output_dir,
            training_config=training_config,
            train_loss_history=train_loss_history,
            valid_loss_history=valid_loss_history,
            global_step=global_step,
            checkpoint_name=archive_checkpoint_name,
        )

        # valid_loss 刷新本次训练最小值时，将当前 latest 原子复制为 best_checkpoint.pt。
        if is_best:
            latest_checkpoint_path = os.path.join(output_dir, "checkpoint_latest.pt")
            best_checkpoint_path = os.path.join(output_dir, "best_checkpoint.pt")
            tmp_best_checkpoint_path = f"{best_checkpoint_path}.tmp"

            shutil.copy2(latest_checkpoint_path, tmp_best_checkpoint_path)
            os.replace(tmp_best_checkpoint_path, best_checkpoint_path)

            print(
                f"saved best checkpoint to {best_checkpoint_path} | "
                f"epoch={epoch} | valid_loss={valid_loss:.4f}"
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ============================================================
    # 7. 训练结束
    # ============================================================
    print(f"best valid loss of this run={best_valid_loss:.4f}")
    print(f"saved best checkpoint to {os.path.join(output_dir, 'best_checkpoint.pt')}")
    print(f"saved tokenizer to {os.path.join(output_dir, 'tokenizer')}")


if __name__ == "__main__":
    main()