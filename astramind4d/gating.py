
import torch
from .core import predictive_entropy

def mc_predict(model, x_short, x_mid, x_long, passes=10):
    outs_b, outs_a = [], []
    for _ in range(passes):
        o = model(x_short, x_mid, x_long, mc_dropout=True)
        outs_b.append(o["behavior_probs"]); outs_a.append(o["action_probs"])
    pb = torch.stack(outs_b).mean(0); pa = torch.stack(outs_a).mean(0)
    Hb = predictive_entropy(pb).mean().item(); Ha = predictive_entropy(pa).mean().item()
    return pb, pa, (Hb+Ha)/2.0

default_tau_by_regime = {"whale":0.42,"institutional":0.44,"algo":0.50,"retail":0.48}

def select_action(acs_out, entropy, tau=0.48):
    K = len(acs_out); B,H,_ = acs_out[0].shape; scores=[]
    for k in range(K):
        U = acs_out[k][:,:,2]; U_mean = U.mean(dim=1); score=U_mean*(1.0-entropy); scores.append(score)
    scores = torch.stack(scores, dim=1); best = torch.argmax(scores, dim=1); gate = (scores.max(dim=1).values >= tau)
    return best, scores, gate
