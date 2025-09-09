
import math, torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TimeScaleBranch(nn.Module):
    def __init__(self, input_size, hidden, num_layers=1, dropout=0.2, heads=4):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_size, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.pos  = PositionalEncoding(hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.enc  = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=hidden*2,
                                       dropout=dropout, activation="gelu", batch_first=True),
            num_layers=1
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        h = self.proj(x); h = self.pos(h)
        h,_ = self.lstm(h); h = self.enc(h)
        z = self.pool(h.transpose(1,2)).squeeze(-1)
        return z

class CrossScaleMixer(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.mixer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden*2,
                                       dropout=dropout, activation="gelu", batch_first=True),
            num_layers=1
        )
    def forward(self, zs):
        Z = torch.stack(zs, dim=1)
        Z = self.mixer(Z)
        return Z.mean(dim=1)

class ACSDecoder(nn.Module):
    def __init__(self, hidden, horizons=(1,3,5), actions=3):
        super().__init__()
        self.horizons = horizons; self.actions = actions
        self.head = nn.Sequential(nn.Linear(hidden+actions, hidden), nn.GELU(), nn.Linear(hidden, len(horizons)*3))
    def forward(self, z):
        outs = []
        B = z.size(0); device = z.device
        for k in range(self.actions):
            a = torch.zeros(B, self.actions, device=device); a[:,k]=1.0
            h = self.head(torch.cat([z,a],dim=-1))
            h = h.view(B, len(self.horizons), 3)
            outs.append(h)
        return outs

class AstraMind4DCore(nn.Module):
    def __init__(self, input_size=20, hidden_size=96, num_layers=2, attn_heads=4, dropout=0.2):
        super().__init__()
        self.br_s = TimeScaleBranch(input_size, hidden_size, num_layers, dropout, attn_heads)
        self.md_s = TimeScaleBranch(input_size, hidden_size, num_layers, dropout, attn_heads)
        self.lg_s = TimeScaleBranch(input_size, hidden_size, num_layers, dropout, attn_heads)
        self.mixer = CrossScaleMixer(hidden_size, dropout)
        self.behavior_head = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size//2, 5))
        self.action_head   = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size//2, 3))
        self.acs = ACSDecoder(hidden_size)
    def forward(self, x_short, x_mid, x_long, mc_dropout=False):
        zs = [self.br_s(x_short), self.md_s(x_mid), self.lg_s(x_long)]
        z  = self.mixer(zs)
        if mc_dropout: self.train()
        b_logits = self.behavior_head(z); a_logits = self.action_head(z)
        b = torch.softmax(b_logits, dim=-1); a = torch.softmax(a_logits, dim=-1)
        acs_out = self.acs(z)
        return {"behavior_probs": b, "action_probs": a, "acs": acs_out, "latent": z}

def predictive_entropy(probs):
    return -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
