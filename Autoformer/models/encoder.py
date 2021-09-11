import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecompLayer(nn.Module):
    def __init__(self):
        super(SeriesDecompLayer, self).__init__()
        self.pooling_layer = nn.AvgPool1d(kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        target_trend = self.pooling_layer(x)
        target_season = x - target_trend
        return target_season, target_trend


class FeedforwardLayer(nn.Module):
    def __init__(self, d_model, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(d_model, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = torch.relu

    def forward(self, x):
        # x = [batch size, seq len, hid dim]
        x = self.dropout(self.activation(self.fc_1(x)))
        # x = [batch size, seq len, pf dim]
        x = self.fc_2(x)
        # x = [batch size, seq len, hid dim]
        return x


class AcEncoderLayer(nn.Module):
    def __init__(self,
                 attention,
                 d_model: int,
                 pf_dim: int = None,
                 dropout: float = 0.1, ) -> None:
        super(AcEncoderLayer, self).__init__()
        pf_dim = pf_dim or d_model * 4
        self.attention = attention
        self.season_block1 = SeriesDecompLayer()
        self.season_block2 = SeriesDecompLayer()
        self.feedforward = FeedforwardLayer(d_model, pf_dim, dropout)

    def forward(self, input_target, attn_mask=None):
        input_ac = self.attention(input_target, input_target, input_target, attn_mask)
        s_enc, s_ent = self.season_block1(input_ac + input_target)
        s_enc2, s_ent2 = self.season_block2(self.feedforward(s_enc) + s_enc)
        return s_enc2


class AcEncoder(nn.Module):
    def __init__(self, encoder_layer):
        super(AcEncoder, self).__init__()
        self.layers = nn.ModuleList(encoder_layer)

    def forward(self, past_target: torch.tensor, attn_mask: torch.tensor = None):
        output = past_target
        for mod in self.layers:
            output = mod(output, attn_mask=attn_mask)

        return output