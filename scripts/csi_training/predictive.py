"""Experimental masked latent CSI prediction, not CSI-JEPA/AM-FM reproduction."""
import argparse
import copy
import hashlib
import json
import os
import stat
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load, save_file

SPEC={'input':[52,190],'axes':['subcarrier','sample_index'],'patch':[4,19],
      'latent':32,'heads':4,'layers':1,'mask_fraction':.4,'ema':.99,
      'variance_weight':.1,'variance_floor':1.,'seed':42,
      'normalization':'train-only per-carrier log-amplitude mean/std',
      'claim':'experimental masked latent prediction, not reproduction or occupancy model'}


def _read(path,limit):
    if not hasattr(os,'O_NOFOLLOW') or not hasattr(os,'O_NONBLOCK'):
        raise ValueError('Secure regular-file reader unavailable on this platform')
    try:
        fd=os.open(Path(path),os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        try:
            info=os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size>limit:
                raise ValueError('Expected bounded regular file')
            with os.fdopen(fd,'rb',closefd=False) as handle:data=handle.read(limit+1)
        finally:
            os.close(fd)
    except OSError as exc:
        raise ValueError('Cannot open verified regular artifact') from exc
    if len(data)>limit:raise ValueError('Artifact exceeds bound')
    return data


def _json(data):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate artifact JSON key')
            result[key]=value
        return result
    def constant(_):raise ValueError('Nonfinite artifact JSON number')
    def number(value):
        result=float(value)
        if not math.isfinite(result):raise ValueError('Nonfinite artifact JSON number')
        return result
    return json.loads(data,object_pairs_hook=pairs,parse_constant=constant,parse_float=number)


def _write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def amplitudes(values):
    x=np.asarray(values)
    if (x.dtype.kind not in 'fiu' or x.ndim!=3 or x.shape[1:]!=(52,190)
            or not 1<=len(x)<=4096 or not np.isfinite(x).all() or (x<0).any()):
        raise ValueError('Expected 1..4096 finite nonnegative amplitude windows N52T190')
    return x


def normalize(values,mean,scale):
    mean,scale=np.asarray(mean),np.asarray(scale)
    if mean.shape!=(52,) or scale.shape!=(52,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or not (scale>0).all():
        raise ValueError('Invalid normalization state')
    x=amplitudes(values).astype(np.float64)
    result=((np.log(np.maximum(x,1e-6))-mean[None,:,None])/scale[None,:,None]).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('Normalization overflow')
    return torch.from_numpy(result)


def split_training(x,identity):
    """Use original training rows only; labels/trial fields never enter features."""
    x=amplitudes(x);ids=np.asarray(identity)
    if (ids.dtype.kind not in 'iu' or ids.shape!=(len(x),4)
            or not np.isin(ids[:,0],[1,2]).all() or not np.isin(ids[:,1],range(1,7)).all()):
        raise ValueError('Only original training room/participant IDs admitted')
    fit=(ids[:,0]==1)&np.isin(ids[:,1],[1,2,3,4])
    held=(ids[:,0]==2)&np.isin(ids[:,1],[5,6])
    return {'train':{'x':x[fit],'groups':ids[fit,:2]},
            'heldout':{'x':x[held],'groups':ids[held,:2]}}


def validate_parts(parts):
    if set(parts)!={'train','heldout'}:raise ValueError('Only group-disjoint inner train/heldout partitions admitted')
    seen=set();groups={}
    for name,part in parts.items():
        x=amplitudes(part['x']);ids=np.asarray(part['groups'])
        if len(x)<2:raise ValueError('Each training/heldout group split requires at least two windows')
        if ids.shape!=(len(x),2) or ids.dtype.kind not in 'iu':raise ValueError('Explicit room/participant groups required')
        groups[name]=[set(ids[:,i]) for i in range(2)]
        for row in x:
            key=hashlib.sha256(np.ascontiguousarray(row,dtype=np.float64).tobytes()).hexdigest()
            if key in seen:raise ValueError('Duplicate recording across admitted population')
            seen.add(key)
    if any(groups['train'][i]&groups['heldout'][i] for i in range(2)):
        raise ValueError('Room or participant group leakage')


class PatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Linear(76,32)
        self.position=nn.Parameter(torch.randn(1,130,32)*.02)
        self.mask_token=nn.Parameter(torch.zeros(1,1,32))
        layer=nn.TransformerEncoderLayer(32,4,dim_feedforward=64,dropout=0.,batch_first=True)
        self.context=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)

    def forward(self,x,mask=None):
        tokens=x.reshape(len(x),13,4,10,19).permute(0,1,3,2,4).reshape(len(x),130,76)
        tokens=self.project(tokens)
        if mask is not None:tokens=torch.where(mask[...,None],self.mask_token,tokens)
        return self.context(tokens+self.position)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=PatchEncoder();self.head=nn.Sequential(nn.Linear(32,32),nn.GELU(),nn.Linear(32,32))
    def forward(self,x,mask):return self.head(self.encoder(x,mask))


def make_mask(count,generator):
    mask=torch.zeros(count,130,dtype=torch.bool)
    indices=torch.rand(count,130,generator=generator).argsort(dim=1)[:,:52]
    return mask.scatter_(1,indices,True)


def objective(student,teacher,x,mask):
    prediction=student(x,mask)
    with torch.no_grad():target=teacher(x)
    latent=student.encoder(x,mask)
    std=torch.sqrt(latent.var(dim=0,unbiased=False)+1e-4)
    variance=F.relu(1.-std).mean()
    loss=F.mse_loss(prediction[mask],target[mask])+.1*variance
    return loss,{'latent_loss':F.mse_loss(prediction[mask],target[mask]).detach(),
                 'sample_std':std.mean().detach(),'variance_penalty':variance.detach()},target


@torch.no_grad()
def ema_update(teacher,student):
    for target,source in zip(teacher.parameters(),student.encoder.parameters()):
        target.mul_(.99).add_(source,alpha=.01)


def _diagnostics(tokens):
    sample_std=tokens.std(dim=0,unbiased=False).mean()
    centered=tokens.mean(1)-tokens.mean((0,1))
    singular=torch.linalg.svdvals(centered)
    power=singular.square();prob=power/power.sum().clamp_min(1e-12)
    rank=torch.exp(-(prob*torch.log(prob.clamp_min(1e-12))).sum()) if power.sum()>0 else torch.tensor(0.)
    return {'sample_std':float(sample_std),'recording_effective_rank':float(rank),
            'collapse_warning':bool(sample_std<.05 or rank<1.5)}


class FrozenEncoder:
    """Verified research encoder; no occupancy labels or automatic adaptation."""
    def __init__(self,directory):
        root=Path(directory);status=_json(_read(root/'status.json',65536))
        if not isinstance(status,dict) or status.get('status')!='complete' or status.get('promotionAuthorized') is not False:raise ValueError('Incomplete research artifact')
        report=_json(_read(root/'metrics.json',1024**2))
        if not isinstance(report,dict) or report.get('status')!='complete' or report.get('spec')!=SPEC or report.get('promotionAuthorized') is not False:raise ValueError('Invalid predictive contract')
        data=_read(root/'model.safetensors',2*1024**2)
        if hashlib.sha256(data).hexdigest()!=report.get('model_sha256'):raise ValueError('Model hash mismatch')
        state=load(data);self.encoder=PatchEncoder().eval().requires_grad_(False)
        expected=self.encoder.state_dict()
        if set(state)!=set(expected)|{'normalization.mean','normalization.scale'}:raise ValueError('Unexpected tensor keys')
        for key,value in expected.items():
            if state[key].shape!=value.shape or state[key].dtype!=value.dtype or not torch.isfinite(state[key]).all():raise ValueError('Invalid encoder tensor')
        self.encoder.load_state_dict({k:state[k] for k in expected},strict=True)
        self.mean=state['normalization.mean'].numpy();self.scale=state['normalization.scale'].numpy()
        if (self.mean.shape!=(52,) or self.scale.shape!=(52,) or self.mean.dtype!=np.float64 or self.scale.dtype!=np.float64
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or not (self.scale>0).all()):raise ValueError('Invalid train normalization')

    def encode(self,values):
        values=amplitudes(values)
        with torch.no_grad():
            result=np.concatenate([self.encoder(normalize(values[i:i+32],self.mean,self.scale)).mean(1).numpy()
                                   for i in range(0,len(values),32)])
        if not np.isfinite(result).all():raise ValueError('Nonfinite frozen embeddings')
        return result

    def new_adapter(self,classes):
        if type(classes) is not int or not 2<=classes<=64:raise ValueError('Adapter classes must be2..64')
        return FrozenAdapter(self,classes)


class FrozenAdapter(nn.Module):
    """Untrained linear head; caller must provide lawful task labels and fresh splits."""
    def __init__(self,bundle,classes):
        super().__init__();self.bundle=bundle;self.head=nn.Linear(32,classes)
    def forward(self,values):return self.head(torch.from_numpy(self.bundle.encode(values)))


def train(parts,provenance,output,epochs=5,batch_size=8,max_seconds=120):
    if type(epochs) is not int or not 1<=epochs<=100 or type(batch_size) is not int or not 2<=batch_size<=64 or type(max_seconds) not in (int,float) or not math.isfinite(max_seconds) or not 1<=max_seconds<=3600:
        raise ValueError('Invalid bounded training configuration')
    source_digest=hashlib.sha256(_read(Path(__file__),1024**2)).hexdigest()
    validate_parts(parts)
    if not isinstance(provenance,dict) or provenance.get('evidence') not in ('SYNTHETIC','PUBLIC_TRAIN_ONLY'):
        raise ValueError('Explicit synthetic or existing public-training lineage required')
    if len(json.dumps(provenance,allow_nan=False))>65536:raise ValueError('Lineage exceeds bound')
    torch.manual_seed(42);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    _write(root/'status.json',{'status':'running','promotionAuthorized':False})
    started=time.monotonic()
    try:
        x=parts['train']['x'].astype(np.float64);logged=np.log(np.maximum(x,1e-6))
        mean=logged.mean(axis=(0,2));scale=np.maximum(logged.std(axis=(0,2)),1e-6)
        training=normalize(x,mean,scale)
        student=Predictor();initial=copy.deepcopy(student).eval();teacher=copy.deepcopy(student.encoder).eval().requires_grad_(False)
        optimizer=torch.optim.AdamW(student.parameters(),lr=1e-3,weight_decay=1e-4)
        generator=torch.Generator().manual_seed(42);history=[]
        for epoch in range(epochs):
            order=torch.randperm(len(training),generator=generator);losses=[]
            for start in range(0,len(order),batch_size):
                if time.monotonic()-started>max_seconds:raise TimeoutError('Budget exceeded between steps; no step preemption guarantee')
                ids=order[start:start+batch_size]
                if len(ids)<2:continue
                mask=make_mask(len(ids),generator);optimizer.zero_grad(set_to_none=True)
                loss,diagnostic,_=objective(student,teacher,training[ids],mask)
                if not torch.isfinite(loss):raise ValueError('Nonfinite pretraining loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1.);optimizer.step();ema_update(teacher,student)
                losses.append(float(loss.detach()))
            history.append({'epoch':epoch+1,'train_objective':float(np.mean(losses))})
        student.eval()
        # Fit constant baseline exclusively on training teacher targets.
        with torch.no_grad():
            target_sum=torch.zeros(32);position_sum=torch.zeros(130,32);count=0;recordings=0
            for offset in range(0,len(training),batch_size):
                if time.monotonic()-started>max_seconds:raise TimeoutError('Budget exceeded during baseline evaluation')
                target=teacher(training[offset:offset+batch_size]);target_sum+=target.sum((0,1));count+=target.numel()//32;position_sum+=target.sum(0);recordings+=len(target)
            constant=target_sum/count;position_constant=position_sum/recordings
        # Final heldout-only evaluation: never used for checkpoints or selection.
        held=normalize(parts['heldout']['x'],mean,scale);evaluations={'trained':[],'untrained':[],'constant':[],'position_constant':[]};latent_rows=[]
        evaluation_rng=torch.Generator().manual_seed(734)
        with torch.no_grad():
            for offset in range(0,len(held),batch_size):
                if time.monotonic()-started>max_seconds:raise TimeoutError('Budget exceeded during heldout evaluation')
                rows=held[offset:offset+batch_size];mask=make_mask(len(rows),evaluation_rng);target=teacher(rows)
                for name,prediction in [('trained',student(rows,mask)),('untrained',initial(rows,mask)),('constant',constant.expand_as(target)),('position_constant',position_constant.expand_as(target))]:
                    evaluations[name].append((float(F.mse_loss(prediction[mask],target[mask])),len(rows)))
                latent_rows.append(student.encoder(rows).cpu())
        losses={name:sum(value*n for value,n in entries)/sum(n for _,n in entries) for name,entries in evaluations.items()}
        state={k:v.detach().cpu().contiguous() for k,v in student.encoder.state_dict().items()}
        state.update({'normalization.mean':torch.from_numpy(mean),'normalization.scale':torch.from_numpy(scale)})
        save_file(state,str(root/'model.safetensors'))
        replay=load(_read(root/'model.safetensors',2*1024**2))
        if any(not torch.equal(state[k],replay[k]) for k in state):raise ValueError('Saved state differs')
        restored=PatchEncoder().eval().requires_grad_(False)
        restored.load_state_dict({k:replay[k] for k in restored.state_dict()},strict=True)
        with torch.no_grad():
            original_embedding=student.encoder(held[:batch_size]).mean(1)
            restored_embedding=restored(held[:batch_size]).mean(1)
        if not torch.equal(original_embedding,restored_embedding):raise ValueError('Reload changes embeddings')
        # Benchmark the exported frozen encoder through its public encode path.
        # Construct from already verified tensors while lifecycle remains running.
        frozen=FrozenEncoder.__new__(FrozenEncoder)
        frozen.encoder=restored;frozen.mean=mean;frozen.scale=scale
        benchmark_input=parts['heldout']['x']
        for i in range(10):frozen.encode(benchmark_input[i%len(benchmark_input):i%len(benchmark_input)+1])
        latency=[]
        for round_id in range(3):
            elapsed=[]
            for i in range(100):
                if time.monotonic()-started>max_seconds:raise TimeoutError('Budget exceeded during inference benchmark')
                j=i%len(benchmark_input);before=time.perf_counter_ns()
                frozen.encode(benchmark_input[j:j+1]);elapsed.append((time.perf_counter_ns()-before)/1e6)
            latency.append({'round':round_id+1,'p50_ms':float(np.percentile(elapsed,50)),
                            'p95_ms':float(np.percentile(elapsed,95))})
        report={'status':'complete','promotionAuthorized':False,'spec':SPEC,'evidence':provenance['evidence'],
                'lineage':provenance,'configuration':{'epochs':epochs,'batch_size':batch_size,'max_seconds':max_seconds},
                'history':history,'heldout_latent_mse':losses,'anti_collapse':_diagnostics(torch.cat(latent_rows)),
                'train_samples':len(training),'heldout_samples':len(held),'normalization_fit':'train_only','reload_embedding_max_error':0.,
                'torch_version':torch.__version__,'numpy_version':np.__version__,
                'model_sha256':hashlib.sha256((root/'model.safetensors').read_bytes()).hexdigest(),
                'source_sha256':source_digest,'inference_latency':{'rounds':latency,'warmup':10,'calls_per_round':100,
                'scope':'CPU batch1 public frozen encode including validation/log normalization/token encoder/mean pooling; excludes acquisition and file IO'},'elapsed_seconds':time.monotonic()-started,
                'scope':'EMA stopped targets; same final teacher target and mask for all baselines. No heldout tuning, original validation/test decoding, activity-to-occupancy relabeling, or task accuracy claim.'}
        if time.monotonic()-started>max_seconds:raise TimeoutError('Budget exceeded before completion')
        if hashlib.sha256(_read(Path(__file__),1024**2)).hexdigest()!=source_digest:raise ValueError('Training source changed during run')
        _write(root/'metrics.json',report);_write(root/'status.json',{'status':'complete','promotionAuthorized':False})
        return report
    except Exception:
        _write(root/'status.json',{'status':'failed','promotionAuthorized':False});raise


def synthetic():
    rng=np.random.default_rng(42);parts={}
    for name,n,room,participant in [('train',24,1,1),('heldout',8,2,5)]:
        phase=rng.uniform(0,6.28,(n,1,1));carrier=rng.normal(0,.2,(n,52,1));time_axis=np.linspace(0,6.28,190)[None,None,:]
        x=np.exp(carrier+.3*np.sin(time_axis+phase)+rng.normal(0,.03,(n,52,190))).astype(np.float32)
        parts[name]={'x':x,'groups':np.tile([room,participant],(n,1))}
    return parts,{'evidence':'SYNTHETIC','source':'deterministic synthetic amplitude fixture, seed42'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--synthetic',action='store_true');group.add_argument('--manifest')
    parser.add_argument('--output',required=True);parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--batch-size',type=int,default=8);parser.add_argument('--max-seconds',type=int,default=120)
    args=parser.parse_args()
    if args.synthetic:parts,provenance=synthetic()
    else:
        from .espfi import _load_partitions
        admitted,meta=_load_partitions(args.manifest,('train',))
        parts=split_training(admitted['train']['x'],admitted['train']['identity'])
        provenance={'evidence':'PUBLIC_TRAIN_ONLY','manifest_sha256':meta['manifest_sha256'],'source_revision':meta['source_revision'],
                    'loaded_partitions':['train'],'inner_fit':'room1 subjects1..4','inner_heldout':'room2 subjects5,6',
                    'unused_training_rows':len(admitted['train']['x'])-sum(len(v['x']) for v in parts.values())}
    print(json.dumps(train(parts,provenance,args.output,args.epochs,args.batch_size,args.max_seconds),indent=2))

if __name__=='__main__':main()
