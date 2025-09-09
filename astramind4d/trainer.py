
import torch, torch.nn.functional as F, torch.optim as optim

def pretrain_with_weak_labels(make_dataset_fn, model, epochs=5, lr=3e-4, device='cpu'):
    model.to(device).train()
    Xs,Xm,Xl,yb,ya = make_dataset_fn()
    Xs=torch.tensor(Xs,dtype=torch.float32,device=device); Xm=torch.tensor(Xm,dtype=torch.float32,device=device); Xl=torch.tensor(Xl,dtype=torch.float32,device=device)
    yb=torch.tensor(yb,dtype=torch.long,device=device); ya=torch.tensor(ya,dtype=torch.long,device=device)
    opt=optim.AdamW(model.parameters(),lr=lr)
    for e in range(epochs):
        opt.zero_grad(); out=model(Xs,Xm,Xl)
        loss_b=F.nll_loss((out['behavior_probs']+1e-8).log(), yb); loss_a=F.nll_loss((out['action_probs']+1e-8).log(), ya)
        loss=loss_b+loss_a; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    model.eval(); return model
