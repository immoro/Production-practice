import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.LazyLinear(d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)
        out = self.out_projection(out)

        return out


def get_roll_V(values, length):
    tmp_list = []
    for i in range(1, length + 1):
        tmp = values.roll(shifts=i, dims=-1)
        tmp_list.append(tmp.unsqueeze(-2))
    tmp_list = torch.cat(tmp_list, axis=-2)
    return tmp_list


class AutoCorrelationLayer(nn.Module):
    def __init__(self, input_dim: int, factor: int = 5
                 ) -> None:
        super(AutoCorrelationLayer, self).__init__()
        self.factor = factor
        self.input_dim = input_dim
        self.q_proj = nn.LazyLinear(1)
        self.k_proj = nn.LazyLinear(1)
        self.v_proj = nn.LazyLinear(1)
        self.out_proj = nn.LazyLinear(input_dim)

    def forward(self,
                queries: torch.tensor,
                keys: torch.tensor,
                values: torch.tensor,
                attn_mask: torch.tensor = None):  # mask 默认为NONE
        B, L_Q, H, n_dim = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        queries = self.q_proj(queries).squeeze()
        keys = self.k_proj(keys).squeeze()
        values = self.v_proj(values).squeeze()

        if L_K < L_Q:
            keys = nn.functional.pad(keys, (0, L_Q - L_K), mode='circular')
            values = nn.functional.pad(values, (0, L_Q - L_K), mode='circular')
        else:
            keys = keys[:, :, -L_Q:]
            values = values[:, :, -L_Q:]

        Q_fft = torch.fft.rfft(queries, 2 * L_Q - 1)
        K_conj_fft = torch.conj(torch.fft.rfft(keys, 2 * L_Q - 1))
        score = torch.fft.irfft(Q_fft * K_conj_fft)
        score = score[:, :, :L_Q]
        score /= L_Q

        k = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # 此处应为向下取整np.floor(np.log())
        weight, indices = torch.topk(score, k, dim=-1, sorted=False)
        weight = torch.softmax(weight, axis=-1)
        indices = L_Q - indices - 1

        V_roll = get_roll_V(values, L_Q)
        V_roll = V_roll[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], indices, :]
        # b: batch_size, h: head, l: seq_length, d: dim
        out = torch.einsum("bhd, bhdl -> bhl", weight, V_roll)
        # out = torch.einsum("bhl, bhld -> bhd", weight, V_roll)[:,:, -L_Q:]
        out = out.unsqueeze(-1)
        # out = self.out_proj(out.unsqueeze(-1))
        return out