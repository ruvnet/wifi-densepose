#!/usr/bin/env python3
"""Rigorous A/B for WiFlow CSI->pose: is the held-out PCK real signal or split leakage?

For a dataset of {csi:[D], kps:17x[x,y,vis]} pairs, train the SAME small MLP under
several train/val SPLITS and report held-out PCK@0.10 vs the mean-pose baseline:

  - chronological_80_20 : last 20% in time (val temporally ADJACENT to train -> leaks
                          via CSI/pose autocorrelation; this is what gave us +9.4)
  - random_80_20        : shuffled (val frames interleaved with train -> MAX leak)
  - blocked_gap         : hold out a contiguous MIDDLE block with a time GAP buffer on
                          each side so val is NOT adjacent to any train frame -> the
                          honest, leakage-controlled test

If the model beats baseline on chronological/random but COLLAPSES to ~baseline on
blocked_gap, the apparent signal was temporal leakage, not generalizable CSI->pose.

Usage (ruvultra venv): python wiflow_ab.py --data ~/wiflow-room/dataset.jsonl
"""
import argparse, json, sys
import numpy as np, torch, torch.nn as nn

def _rec(r, X, Y, V, B):
    X.append(r["csi"]); kp=r["kps"]
    if kp and isinstance(kp[0], (list,tuple)):       # 17 x [x,y(,vis)]
        Y.append([c for k in kp for c in (k[0],k[1])]); V.append([(k[2] if len(k)>2 else 1.0) for k in kp])
    else:                                            # flat 34 (browser export, no vis)
        Y.append(list(kp)); V.append([1.0]*17)
    B.append(r.get("bucket"))

def load(path):
    X,Y,V,B=[],[],[],[]
    txt=open(path).read().strip()
    if txt[:1] in "[{":                               # JSON (browser export: dict{samples:[]} or bare array)
        d=json.loads(txt)
        rows = d if isinstance(d,list) else d.get("samples", d.get("data", []))
        for r in rows: _rec(r,X,Y,V,B)
    else:                                             # JSONL (python capture)
        for line in txt.splitlines():
            if line.strip(): _rec(json.loads(line),X,Y,V,B)
    return np.array(X,np.float32), np.array(Y,np.float32), np.array(V,np.float32), B

class Net(nn.Module):
    def __init__(s,din,dout):
        super().__init__()
        s.n=nn.Sequential(nn.Linear(din,384),nn.ReLU(),nn.Dropout(.35),
                          nn.Linear(384,192),nn.ReLU(),nn.Dropout(.35),
                          nn.Linear(192,96),nn.ReLU(),nn.Linear(96,dout),nn.Sigmoid())
    def forward(s,x): return s.n(x)

def pck(pred,gt,vis,thr=0.10):
    p=pred.reshape(-1,17,2); g=gt.reshape(-1,17,2)
    d=np.linalg.norm(p-g,axis=2); m=vis>0.5
    return float((d[m]<thr).mean()) if m.any() else 0.0

def split_idx(n, kind, B=None):
    idx=np.arange(n)
    if kind=="chronological_80_20":
        c=int(n*.8); return idx[:c], idx[c:]
    if kind=="random_80_20":
        rng=np.random.default_rng(0); p=rng.permutation(n); c=int(n*.8); return p[:c], p[c:]
    if kind=="blocked_gap":
        # val = contiguous middle 20%; a WIDE 10% time gap each side guarantees no train
        # frame is temporally adjacent to a val frame (kills frame-autocorrelation leakage).
        v0=int(n*.4); v1=int(n*.6); gap=int(n*.10)
        val=idx[v0:v1]; train=np.concatenate([idx[:max(0,v0-gap)], idx[min(n,v1+gap):]])
        return train, val
    if kind=="grouped_bucket":
        # hold out ENTIRE activity buckets -> val poses/activities never seen in train.
        # the strictest leakage-free test (only when bucket labels exist).
        b=np.array([x if x is not None else -1 for x in B])
        uniq=[u for u in sorted(set(b.tolist())) if u!=-1]
        if len(uniq)<3: raise ValueError("too few buckets")
        hold=set(uniq[::max(1,len(uniq)//3)][:max(1,len(uniq)//3)])  # ~1/3 of activities held out
        val=idx[np.isin(b,list(hold))]; train=idx[~np.isin(b,list(hold))]
        return train, val
    raise ValueError(kind)

def run(X,Y,V,tr,va,epochs=250,seed=0):
    torch.manual_seed(seed); np.random.seed(seed)   # seed weight init + batch shuffle
    dev="cuda" if torch.cuda.is_available() else "cpu"
    mu,sd=X[tr].mean(0),X[tr].std(0)+1e-6
    Xtr=torch.tensor((X[tr]-mu)/sd).to(dev); Ytr=torch.tensor(Y[tr]).to(dev)
    Xva=torch.tensor((X[va]-mu)/sd).to(dev)
    net=Net(X.shape[1],Y.shape[1]).to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4); lf=nn.MSELoss()
    best=(1e9,None)
    for ep in range(epochs):
        net.train(); perm=torch.randperm(len(Xtr),device=dev)
        for i in range(0,len(Xtr),64):
            j=perm[i:i+64]; opt.zero_grad(); loss=lf(net(Xtr[j]),Ytr[j]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): pv=net(Xva).cpu().numpy()
        vl=float(((pv-Y[va])**2).mean())
        if vl<best[0]: best=(vl,pv)
    base=np.tile(Y[tr].mean(0),(len(va),1))
    return pck(best[1],Y[va],V[va]), pck(base,Y[va],V[va])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True)
    ap.add_argument("--epochs",type=int,default=250); ap.add_argument("--seeds",type=int,default=3)
    a=ap.parse_args()
    X,Y,V,B=load(a.data); n=len(X)
    has_buckets=any(x is not None for x in B)
    print(f"[ab] {n} samples, X={X.shape}, buckets={'yes' if has_buckets else 'no'}, "
          f"seeds={a.seeds}, epochs={a.epochs}\n")
    print(f"{'split':<22}{'model PCK@0.10':>16}{'baseline':>11}{'delta (mean±sd)':>20}   verdict")
    print("-"*86)
    splits=["chronological_80_20","random_80_20","blocked_gap"]+(["grouped_bucket"] if has_buckets else [])
    for kind in splits:
        try:
            tr,va=split_idx(n,kind,B)
            ms=[]; bs=[]
            for s in range(a.seeds):
                m,b=run(X,Y,V,tr,va,a.epochs,seed=s); ms.append(m); bs.append(b)
            ms=np.array(ms)*100; bs=np.array(bs)*100; ds=ms-bs
            dm,dsd=ds.mean(),ds.std()
            # REAL only if the mean delta minus 1 sd still clears the 1.5pp threshold (robust to seed variance)
            verdict = "REAL signal" if dm-dsd>1.5 else ("weak/uncertain" if dm>1.5 else "no signal (==baseline)")
            print(f"{kind:<22}{ms.mean():>13.1f}±{ms.std():>3.1f}{bs.mean():>10.1f}%{dm:>+12.1f}±{dsd:>4.1f}pp   {verdict}")
        except Exception as e:
            print(f"{kind:<22}  skipped: {e}")
    print(f"\nmean±sd over {a.seeds} seeds (weight init + batch order). blocked_gap = 10% time gap each")
    print("side; grouped_bucket holds out ENTIRE activities (strictest). If only the LEAKY splits")
    print("(chronological/random) beat baseline, the apparent signal is leakage, not generalizable pose.")

if __name__=="__main__": main()
