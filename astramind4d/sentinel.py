
import numpy as np

class MetaSentinel:
    def __init__(self, window_ref=256, window_cur=128, d_thresh=0.18, H_max=0.45):
        self.window_ref=window_ref; self.window_cur=window_cur; self.d_thresh=d_thresh; self.H_max=H_max; self.buffer=[]
    def _empirical_cdf(self, x):
        xs=np.sort(x); n=len(xs)
        def cdf(v): return (xs.searchsorted(v,side='right'))/n
        return cdf,xs
    def ks_distance(self,a,b):
        a=np.asarray(a); b=np.asarray(b); ca,xa=self._empirical_cdf(a); cb,xb=self._empirical_cdf(b)
        xs=np.unique(np.concatenate([xa,xb])); da=np.array([ca(x) for x in xs]); db=np.array([cb(x) for x in xs])
        return float(np.max(np.abs(da-db)))
    def update(self,value,entropy):
        self.buffer.append(float(value))
        if len(self.buffer)>(self.window_ref+self.window_cur):
            self.buffer=self.buffer[-(self.window_ref+self.window_cur):]
        ref=self.buffer[:self.window_ref] if len(self.buffer)>=(self.window_ref+self.window_cur) else None
        cur=self.buffer[-self.window_cur:] if len(self.buffer)>=self.window_cur else None
        drift=False
        if ref is not None and cur is not None:
            D=self.ks_distance(ref,cur); drift=D>=self.d_thresh
        unsafe_entropy=entropy>=self.H_max
        return {"drift":drift,"high_entropy":unsafe_entropy,"safe":(not drift) and (not unsafe_entropy)}
