import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

from preprocess import *

import math

class Conv1dStack(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1, dilation=1):
        super(Conv1dStack, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )
        self.res = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        h = self.res(x)
        return x + h


class Conv2dStack(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1, dilation=1):
        super(Conv2dStack, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )
        self.res = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        h = self.res(x)
        return x + h


class SeqEncoder(nn.Module):
    def __init__(self, in_dim: int):
        super(SeqEncoder, self).__init__()
        self.conv0 = Conv1dStack(in_dim, 128, 3, padding=1)
        self.conv1 = Conv1dStack(128, 64, 6, padding=5, dilation=2)
        self.conv2 = Conv1dStack(64, 32, 15, padding=7, dilation=1)
        self.conv3 = Conv1dStack(32, 32, 30, padding=29, dilation=2)

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        # x = x.permute(0, 2, 1).contiguous()
        # BATCH x 256 x seq_length
        return x


class BppAttn(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super(BppAttn, self).__init__()
        self.conv0 = Conv1dStack(in_channel, out_channel, 3, padding=1)
        self.bpp_conv = Conv2dStack(5, out_channel)

    def forward(self, x, bpp):
        x = self.conv0(x)
        bpp = self.bpp_conv(bpp)
        # BATCH x C x SEQ x SEQ
        # BATCH x C x SEQ
        x = torch.matmul(bpp, x.unsqueeze(-1))
        return x.squeeze(-1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerWrapper(nn.Module):
    def __init__(self, dmodel=256, nhead=8, num_layers=2):
        super(TransformerWrapper, self).__init__()
        self.pos_encoder = PositionalEncoding(256)
        encoder_layer = TransformerEncoderLayer(d_model=dmodel, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.pos_emb = PositionalEncoding(dmodel)

    def flatten_parameters(self):
        pass

    def forward(self, x):
        x = x.permute((1, 0, 2)).contiguous()
        x = self.pos_emb(x)
        x = self.transformer_encoder(x)
        x = x.permute((1, 0, 2)).contiguous()
        return x, None


class RnnLayers(nn.Module):
    def __init__(self, dmodel, dropout=0.3, transformer_layers: int = 2):
        super(RnnLayers, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.rnn0 = TransformerWrapper(dmodel, nhead=8, num_layers=transformer_layers)
        self.rnn1 = nn.LSTM(dmodel, dmodel // 2, batch_first=True, num_layers=1, bidirectional=True)
        self.rnn2 = nn.GRU(dmodel, dmodel // 2, batch_first=True, num_layers=1, bidirectional=True)

    def forward(self, x):
        self.rnn0.flatten_parameters()
        x, _ = self.rnn0(x)
        if self.rnn1 is not None:
            self.rnn1.flatten_parameters()
            x = self.dropout(x)
            x, _ = self.rnn1(x)
        if self.rnn2 is not None:
            self.rnn2.flatten_parameters()
            x = self.dropout(x)
            x, _ = self.rnn2(x)
        return x

    
class BaseAttnModel(nn.Module):
    def __init__(self, transformer_layers: int = 2):
        super(BaseAttnModel, self).__init__()
        self.linear0 = nn.Linear(14 + 3, 1)
        self.seq_encoder_x = SeqEncoder(18)
        self.attn = BppAttn(256, 128)
        self.seq_encoder_bpp = SeqEncoder(128)
        self.seq = RnnLayers(256 * 2, dropout=0.3,
                             transformer_layers=transformer_layers)

    def forward(self, x, bpp):
        bpp_features = get_bpp_feature(bpp[:, :, :, 0].float())
        x = torch.cat([x] + bpp_features, dim=-1)
        learned = self.linear0(x)
        x = torch.cat([x, learned], dim=-1)
        x = x.permute(0, 2, 1).contiguous().float()
        # BATCH x 18 x seq_len
        bpp = bpp.permute([0, 3, 1, 2]).contiguous().float()
        # BATCH x 5 x seq_len x seq_len
        x = self.seq_encoder_x(x)
        # BATCH x 256 x seq_len
        bpp = self.attn(x, bpp)
        bpp = self.seq_encoder_bpp(bpp)
        # BATCH x 256 x seq_len
        x = x.permute(0, 2, 1).contiguous()
        # BATCH x seq_len x 256
        bpp = bpp.permute(0, 2, 1).contiguous()
        # BATCH x seq_len x 256
        x = torch.cat([x, bpp], dim=2)
        # BATCH x seq_len x 512
        x = self.seq(x)
        return x


class AEModel(nn.Module):
    def __init__(self, transformer_layers: int = 2):
        super(AEModel, self).__init__()
        self.seq = BaseAttnModel(transformer_layers=transformer_layers)
        self.linear = nn.Sequential(
            nn.Linear(256 * 2, 14),
            nn.Sigmoid(),
        )

    def forward(self, x, bpp):
        x = self.seq(x, bpp)
        x = F.dropout(x, p=0.3)
        x = self.linear(x)
        return x


class FromAeModel(nn.Module):
    def __init__(self, seq, pred_len=68, dmodel: int = 256):
        super(FromAeModel, self).__init__()
        self.seq = seq
        self.pred_len = pred_len
        self.linear = nn.Sequential(
            nn.Linear(dmodel * 2, len(target_cols)),
        )

    def forward(self, x, bpp):
        x = self.seq(x, bpp)
        x = self.linear(x)
        x = x[:, :self.pred_len]
        return x