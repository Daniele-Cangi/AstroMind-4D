
import torch
from astramind4d import AstraMind4DCore, mc_predict, select_action, default_tau_by_regime
from astramind4d.losses import physics_loss

B,T,F=4,64,20
x_short=torch.randn(B,T,F); x_mid=torch.randn(B,T,F); x_long=torch.randn(B,T,F)
model=AstraMind4DCore()
out=model(x_short,x_mid,x_long)
pb,pa,H=mc_predict(model,x_short,x_mid,x_long,passes=6)
best,scores,gate=select_action(out["acs"],H,tau=default_tau_by_regime["retail"])
print("Entropy:",H,"Best:",best.tolist(),"Gate:",gate.tolist())
batch={"acs":out["acs"]}; print("Physics loss:",float(physics_loss(batch)))
