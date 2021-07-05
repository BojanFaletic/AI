import torch.nn as nn
from copy import deepcopy
from .attention import MultiHeadAttention
from .positional_encoding import PositionalEncoding


def feed_forward(dim_input: int = 512, dim_feedback: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedback),
        nn.ReLU(),
        nn.Linear(dim_feedback, dim_input)
    )


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = feed_forward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm2_(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.att_1 = MultiHeadAttention(heads, d_model)
        self.att_2 = MultiHeadAttention(heads, d_model)
        self.ff = feed_forward(d_model)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.att_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.att_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))


def get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, e_outputs, src_mask, trg_mask)
        return self.norm(x)
