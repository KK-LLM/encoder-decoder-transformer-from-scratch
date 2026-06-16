import os

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset, DataLoader

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# 基于脚本文件位置解析项目根目录，确保从任意位置运行均可正确找到 tokenizer 和数据。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# 初始占位，后续会根据 tokenizer 自动更新。
PAD_ID = 0
BOS_ID = 101
EOS_ID = 102


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
        # x: [..., d_model]
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


class Generator(nn.Module):
    """
        decoder_out -> Linear -> logits
        [B, S, d_model] -> [B, S, vocab]
        loss = CrossEntropyLoss(logits, target)
    """

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

# 将训练样本转换成 src/tgt token ids。
class TranslationDataset(TorchDataset):
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
            translation = item["translation"]

            src_text = translation[self.source_lang]
            tgt_text = translation[self.target_lang]

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

    def __getitem__(self, index):
        if isinstance(index, list):
            return [self.samples[i] for i in index]
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
    step: int,              # 当前训练步数；通常每执行一次 optimizer.step() / scheduler.step()，step 增加 1
    d_model: int,           # Transformer 的 hidden size / 模型维度，用于按 d_model^(-0.5) 缩放学习率
    warmup_steps: int,      # warmup 阶段的步数；在此之前学习率逐步升高，之后开始衰减
    factor: float = 1.0     # 学习率整体缩放系数；用于整体放大或缩小 Noam 学习率
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


def save_training_checkpoint(
    model: EncoderDecoderTransformer,
    tokenizer,
    epoch: int,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    cpu_state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": cpu_state_dict,
        },
        checkpoint_path,
    )

    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

    print(f"saved checkpoint to {checkpoint_path}")


def main() -> None:
    device = get_best_device()

    # ============================================================
    # 1. 基础配置
    # ============================================================
    tokenizer_name = os.path.join(_PROJECT_ROOT, "tokenizer")
    new_tokenizer_vocab_size = 16000

    source_lang = "en"
    target_lang = "zh"

    max_src_len = 96
    max_tgt_len = 96

    d_model = 512
    d_ff = 2048
    num_heads = 8
    num_layers = 8
    dropout = 0.1

    batch_size = 280

    num_epochs = 48
    warmup_steps = 3000
    label_smoothing = 0.1

    max_train_samples = 500000
    max_valid_samples = 2000

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16

    num_workers = min(8, os.cpu_count() or 1)
    pin_memory = device.type == "cuda"

    output_dir = "minimal_transformer_en_zh_opus_outputs"

    print(f"device={device}")

    if device.type == "cuda":
        print(f"Using CUDA acceleration: {torch.cuda.get_device_name(0)}")
        print(f"AMP enabled={use_amp}, dtype={amp_dtype}")
    else:
        print("CUDA is not available. Training will be much slower.")

    # ============================================================
    # 2. 加载真实中英文翻译数据集
    # ============================================================
    train_ds = Dataset.load_from_disk(os.path.join(_PROJECT_ROOT, "data", "opus100_en_zh_local", "train"))
    valid_ds = Dataset.load_from_disk(os.path.join(_PROJECT_ROOT, "data", "opus100_en_zh_local", "validation"))
    raw_dataset = DatasetDict({"train": train_ds, "validation": valid_ds})

    train_raw = raw_dataset["train"]
    valid_raw = raw_dataset["validation"]

    if max_train_samples is not None:
        train_raw = train_raw.select(range(min(max_train_samples, len(train_raw))))

    if max_valid_samples is not None:
        valid_raw = valid_raw.select(range(min(max_valid_samples, len(valid_raw))))

    print(f"train raw samples={len(train_raw)}")
    print(f"valid raw samples={len(valid_raw)}")

    # ============================================================
    # 3. 基于中英文训练语料重新训练一个小词表 tokenizer
    # ============================================================
    base_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def tokenizer_text_iterator(hf_dataset, batch_size_for_tokenizer: int = 1000):
        for start in range(0, len(hf_dataset), batch_size_for_tokenizer):
            end = min(start + batch_size_for_tokenizer, len(hf_dataset))
            batch = hf_dataset[start:end]["translation"]

            texts = []
            for item in batch:
                src_text = item[source_lang]
                tgt_text = item[target_lang]

                if isinstance(src_text, str) and src_text.strip():
                    texts.append(src_text)

                if isinstance(tgt_text, str) and tgt_text.strip():
                    texts.append(tgt_text)

            yield texts

    tokenizer = base_tokenizer.train_new_from_iterator(
        tokenizer_text_iterator(train_raw),
        vocab_size=new_tokenizer_vocab_size,
    )

    global PAD_ID, BOS_ID, EOS_ID
    PAD_ID = tokenizer.pad_token_id
    BOS_ID = tokenizer.cls_token_id
    EOS_ID = tokenizer.sep_token_id

    assert PAD_ID is not None, "PAD_ID is None. Please check tokenizer.pad_token."
    assert BOS_ID is not None, "BOS_ID is None. Please check tokenizer.cls_token."
    assert EOS_ID is not None, "EOS_ID is None. Please check tokenizer.sep_token."

    vocab_size = len(tokenizer)

    print(f"base tokenizer={tokenizer_name}")
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

    print(f"train dataset size={len(train_dataset)}")
    print(f"valid dataset size={len(valid_dataset)}")
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

    # ============================================================
    # 6. 训练与验证
    # ============================================================
    example_texts = [
		"I love machine-learning.",
        "Please close the door.",
        "I need a cup of coffee.",
        "She forgot to bring her notebook.",
        "The meeting starts at nine tomorrow.",
        "This is a small Transformer model.",
        "We need more clean data.",
        "I'll be there in about ten minutes, just wait for me.",
        "The encoder reads the English sentence.",
        "Save the checkpoint after each epoch.",
		"Let me know if there's anything else I can help you with.",
		"To be honest, I'd rather stay home than go out tonight."
    ]

    for epoch in range(1, num_epochs + 1):
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

        print(
            f"epoch={epoch} | "
            f"train_loss={train_loss:.4f} | "
            f"valid_loss={valid_loss:.4f} | "
            f"lr={current_lr:.8f}"
        )

        for example_text in example_texts:
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

        # 每 4 个 epoch 保存一次 checkpoint。
        if epoch % 4 == 0:
            save_training_checkpoint(
                model=model,
                tokenizer=tokenizer,
                epoch=epoch,
                output_dir=output_dir,
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ============================================================
    # 7. 保存最终模型参数和 tokenizer
    # ============================================================
    os.makedirs(output_dir, exist_ok=True)

    cpu_state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    final_model_path = os.path.join(output_dir, "final_model.pt")

    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": cpu_state_dict,
        },
        final_model_path,
    )

    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

    print(f"saved final model to {final_model_path}")
    print(f"saved tokenizer to {os.path.join(output_dir, 'tokenizer')}")


if __name__ == "__main__":
    main()