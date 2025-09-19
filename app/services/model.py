import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
 
    def __init__(self, d_model, dropout=0.1, max_len=70):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SignLanguageTransformer(nn.Module):
   
    def __init__(self, num_classes, input_features, d_model, nhead, num_encoder_layers, dropout, max_len=70):
        super(SignLanguageTransformer, self).__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src_key_padding_mask = (src.sum(dim=2) == 0)
        src = self.input_proj(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = output.mean(dim=1)
        return self.classifier(output)

