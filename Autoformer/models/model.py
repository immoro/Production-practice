import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import AcEncoder, AcEncoderLayer, SeriesDecompLayer
from models.decoder import AcDecoder, AcDecoderLayer
from models.attn import AutoCorrelationLayer, AttentionLayer
from models.embed import DataEmbedding


class AutoformerNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 context_length: int,
                 prediction_length: int,
                 factor: int = 5,
                 d_model: int = 512,
                 n_heads: int = 8,
                 e_layers: int = 3,
                 d_layers: int = 2,
                 pf_dim: int = 1024,
                 freq: str = 'D',
                 embed: str = 'timeF',
                 dropout: float = 0.1,
                 device: str = 'cuda:0',
                 ):
        super(AutoformerNetwork, self).__init__()

        self.input_dim = input_dim
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.factor = factor
        self.d_model = d_model
        self.n_heads = n_heads
        self.encoder_layers = e_layers
        self.decoder_layers = d_layers
        self.pf_dim = pf_dim
        self.freq = freq
        self.embed = embed
        self.dropout = dropout
        self.sub_dim = d_model // n_heads
        self.device = device

        self.season_block = SeriesDecompLayer()
        self.enc_embedder = DataEmbedding(c_in=input_dim, d_model=d_model,
                                          freq=freq, embed_type=embed,
                                          dropout=dropout)
        self.dec_embebdder = DataEmbedding(c_in=input_dim, d_model=d_model,
                                           freq=freq, embed_type=embed,
                                           dropout=dropout)
        self.encoder = AcEncoder([AcEncoderLayer(
            AttentionLayer(
                attention=AutoCorrelationLayer(factor=factor,
                                               input_dim=self.sub_dim),
                d_model=d_model,
                n_heads=n_heads),
            d_model=d_model,
            pf_dim=pf_dim,
            dropout=dropout)
            for _ in range(e_layers)])
        self.decoder = AcDecoder([AcDecoderLayer(
            attention=AttentionLayer(AutoCorrelationLayer(factor=factor,
                                                          input_dim=self.sub_dim),
                                     d_model=d_model,
                                     n_heads=n_heads)
            , cross_attention=AttentionLayer(AutoCorrelationLayer(factor=factor,
                                                                  input_dim=self.sub_dim),
                                             d_model=d_model,
                                             n_heads=n_heads),
            input_dim=input_dim,
            d_model=d_model,
            pf_dim=pf_dim,
            dropout=dropout)
            for _ in range(d_layers)])
        self.season_output = nn.LazyLinear(input_dim)
        self.trend_output = nn.LazyLinear(input_dim)

    def forward(self,
                past_target: torch.Tensor,
                past_time_feat: torch.Tensor,
                future_time_feat: torch.Tensor,
                ) -> torch.Tensor:
        enc_input = self.enc_embedder(past_target, past_time_feat)
        enc_input = self.encoder(enc_input)

        past_season, past_trend = self.season_block(past_target[:, -self.context_length // 2:, :])
        past_mean = torch.mean(past_target, axis=1)
        season_init = torch.zeros([past_target.shape[0],
                                     self.prediction_length,
                                     past_target.shape[-1]]).to(self.device)
        season_init = torch.cat([past_season, season_init], dim=1)

        trend_init = past_mean.unsqueeze(1).expand(-1, self.prediction_length, -1)
        trend_init = torch.cat([past_trend, trend_init], dim=1)
        future_pad_time_feat = past_time_feat[:, -self.context_length // 2:, :]
        future_time_feat = torch.cat([future_pad_time_feat, future_time_feat], dim=1)

        dec_season = self.dec_embebdder(season_init, future_time_feat)
        dec_season, dec_trend = self.decoder(dec_season, trend_init, enc_input)

        out = self.season_output(dec_season) + self.trend_output(dec_trend)
        out = out[:, -self.prediction_length:, :].squeeze()
        return out