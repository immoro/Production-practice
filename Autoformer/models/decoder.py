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

class AcDecoderLayer(nn.Module):
    def __init__(self, attention, cross_attention, input_dim, d_model, pf_dim, dropout: float = 0.1):
        super(AcDecoderLayer, self).__init__()
        pf_dim = pf_dim or d_model * 4
        self.attention = attention
        self.cross_attention = cross_attention
        self.d_model = d_model
        self.season_block1 = SeriesDecompLayer()
        self.season_block2 = SeriesDecompLayer()
        self.season_block3 = SeriesDecompLayer()

        self.proj_1 = nn.LazyLinear(input_dim)
        self.proj_2 = nn.LazyLinear(input_dim)
        self.proj_3 = nn.LazyLinear(input_dim)
        self.feedforward = FeedforwardLayer(d_model, pf_dim, dropout)

    def forward(self, dec_season, dec_trend, enc_input):
        out_ac = self.attention(dec_season, dec_season, dec_season, None)
        dec_s1, dec_t1 = self.season_block1(out_ac + dec_season)
        cross_ac = self.cross_attention(dec_s1, enc_input, enc_input, None)
        dec_s2, dec_t2 = self.season_block2(cross_ac + dec_s1)
        dec_s3, dec_t3 = self.season_block3(self.feedforward(dec_s2) + dec_s2)

        dec_season = dec_s3
        dec_trend = dec_trend + self.proj_1(dec_t1) + self.proj_2(dec_t2) + self.proj_3(dec_t3)
        return dec_season, dec_trend


class AcDecoder(nn.Module):
    def __init__(self, decoder_layer):
        super(AcDecoder, self).__init__()
        self.layers = nn.ModuleList(decoder_layer)

    def forward(self, dec_season, dec_trend, enc_input):
        for mod in self.layers:
            dec_season, dec_trend = mod(dec_season, dec_trend, enc_input)
        return dec_season, dec_trend