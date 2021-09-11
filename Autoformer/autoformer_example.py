import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)#不理解此处第三维是什么
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[str.lower(freq)]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        #不理解此处对embed_type的处理
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

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
    for i in range(1,length+1):
        tmp = values.roll(shifts = i, dims = -1)
        tmp_list.append(tmp.unsqueeze(-2))
    tmp_list = torch.cat(tmp_list, axis = -2)
    return tmp_list


class AutoCorrelationLayer(nn.Module):
    def __init__(self, input_dim: int, factor: int = 5
                )-> None:
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
                attn_mask: torch.tensor = None):#mask 默认为NONE
        B, L_Q, H, n_dim = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)
        
        queries = self.q_proj(queries).squeeze()
        keys = self.k_proj(keys).squeeze()
        values = self.v_proj(values).squeeze()
        
        if L_K < L_Q:
            keys = nn.functional.pad(keys, (0, L_Q - L_K), mode = 'circular')
            values = nn.functional.pad(values, (0, L_Q - L_K), mode = 'circular')
        else:
            keys = keys[:, :,-L_Q:]
            values = values[:, :,-L_Q:]
            
        
        
        Q_fft = torch.fft.rfft(queries, 2 * L_Q - 1)
        K_conj_fft = torch.conj(torch.fft.rfft(keys, 2 * L_Q - 1))
        score = torch.fft.irfft(Q_fft * K_conj_fft)
        score = score[:,:,:L_Q]
        score /= L_Q
        
        k = self.factor * np.ceil(np.log(L_Q)).astype('int').item()#此处应为向下取整np.floor(np.log())
        weight, indices = torch.topk(score, k, dim = -1, sorted=False)
        weight = torch.softmax(weight, axis = -1)
        indices = L_Q - indices - 1

        V_roll = get_roll_V(values, L_Q)
        V_roll = V_roll[torch.arange(B)[:, None,None], torch.arange(H)[None, :, None], indices, :]
        #b: batch_size, h: head, l: seq_length, d: dim
        out = torch.einsum("bhd, bhdl -> bhl", weight, V_roll)
        # out = torch.einsum("bhl, bhld -> bhd", weight, V_roll)[:,:, -L_Q:]
        out = out.unsqueeze(-1)
        # out = self.out_proj(out.unsqueeze(-1))
        return out
    
#对于padding，考虑根据topk时序进行padding
class SeriesDecompLayer(nn.Module):
    def __init__(self):
        super(SeriesDecompLayer, self).__init__()
        self.pooling_layer = nn.AvgPool1d(kernel_size =3, padding=1 , stride=1)
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
        #x = [batch size, seq len, hid dim]
        x = self.dropout(self.activation(self.fc_1(x)))
        #x = [batch size, seq len, pf dim]
        x = self.fc_2(x)
        #x = [batch size, seq len, hid dim]
        return x
    

class AcEncoderLayer(nn.Module):
    def __init__(self, 
                 attention, 
                 d_model: int,
                 pf_dim:int = None, 
                 dropout: float = 0.1,) -> None:
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
            output = mod(output, attn_mask = attn_mask)
        
        return output
    
    
class AcDecoderLayer(nn.Module):
    def __init__(self, attention, cross_attention,input_dim, d_model, pf_dim, dropout: float = 0.1):
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


class AutoformerNetwork(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 context_length: int, 
                 prediction_length: int,
                 factor: int = 5,
                 d_model: int = 512, 
                 n_heads: int = 8,
                 encoder_layers: int = 3, 
                 decoder_layers: int = 2,
                 pf_dim: int = 1024,
                 freq: str = 'D', 
                 embed:str = 'timeF',
                 dropout: float = 0.1,
                 device: str = 'cuda:0',
                ):
        super(AutoformerNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.factor =factor
        self.d_model = d_model
        self.n_heads = n_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.pf_dim = pf_dim
        self.freq = freq
        self.embed = embed
        self.dropout = dropout
        self.sub_dim = d_model//n_heads
        self.device = device
        
        self.season_block = SeriesDecompLayer()
        self.enc_embedder = DataEmbedding(c_in=input_dim, d_model=d_model, 
                                      freq=freq, embed_type=embed, 
                                      dropout = dropout)
        self.dec_embebdder = DataEmbedding(c_in=input_dim, d_model=d_model, 
                                       freq=freq, embed_type=embed, 
                                       dropout = dropout)
        self.encoder = AcEncoder([AcEncoderLayer(
                            AttentionLayer(
                                attention=AutoCorrelationLayer(factor = factor, 
                                                     input_dim = self.sub_dim),
                                d_model = d_model,
                                n_heads = n_heads), 
                                d_model = d_model, 
                                pf_dim = pf_dim, 
                                dropout = dropout) 
                                for _  in range(encoder_layers)])
        self.decoder = AcDecoder([AcDecoderLayer(
                                attention = AttentionLayer(AutoCorrelationLayer(factor = factor,
                                                    input_dim = self.sub_dim),
                                                    d_model = d_model,
                                                    n_heads = n_heads)
                                    ,cross_attention = AttentionLayer(AutoCorrelationLayer(factor = factor,
                                                    input_dim = self.sub_dim),
                                                    d_model = d_model,
                                                    n_heads = n_heads), 
                                    input_dim = input_dim, 
                                    d_model = d_model, 
                                    pf_dim = pf_dim, 
                                    dropout = dropout) 
                                 for _  in range(decoder_layers)])
        self.season_output = nn.LazyLinear(input_dim)
        self.trend_output = nn.LazyLinear(input_dim)
    def forward(self, 
                past_target: torch.Tensor,
                past_time_feat: torch.Tensor,
                future_time_feat: torch.Tensor,
                ) -> torch.Tensor:
        enc_input = self.enc_embedder(past_target, past_time_feat)
        enc_input = self.encoder(enc_input)
        
        past_season, past_trend = self.season_block(past_target[:, -self.context_length//2:, :])
        past_mean = torch.mean(past_target, axis = 1)
        future_target = torch.zeros([past_target.shape[0], 
                                  self.prediction_length, 
                                  past_target.shape[-1]]).to(self.device)
        future_target = torch.cat([past_season, future_target], dim=1)
        
        dec_trend = past_mean.unsqueeze(1).expand(-1, self.prediction_length, -1)
        dec_trend = torch.cat([past_trend, dec_trend], dim=1)
        future_pad_time_feat = past_time_feat[:, -self.context_length//2:, :]
        future_time_feat = torch.cat([future_pad_time_feat, future_time_feat], dim = 1)
        
        dec_season = self.dec_embebdder(future_target, future_time_feat)
        dec_season, dec_trend = self.decoder(dec_season, dec_trend, enc_input)
        
        out = self.season_output(dec_season) + self.trend_output(dec_trend)
        out = out[:, -self.prediction_length:, :].squeeze()
        return out