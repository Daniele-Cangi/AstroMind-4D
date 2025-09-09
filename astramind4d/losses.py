
import torch

def physics_loss(batch, max_move_per_step=0.2, volume_smooth_lambda=0.0):
    acs = batch["acs"]; penalty=0.0
    for k in range(len(acs)):
        dy = acs[k][:,:,0]; penalty += torch.relu(torch.abs(dy)-max_move_per_step).mean()
    if volume_smooth_lambda and "vol" in batch:
        v = batch["vol"].float(); dv = (v[:,1:]-v[:,:-1]).abs().mean(); penalty += volume_smooth_lambda*dv
    return penalty

def variance_of_entropy(entropies):
    if not torch.is_tensor(entropies):
        entropies = torch.tensor(entropies, dtype=torch.float32)
    return torch.var(entropies)
