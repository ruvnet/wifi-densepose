"""Fixed two-arm masked-CSI representation experiment; VICReg-inspired, not reproduction."""
import argparse
import copy
import hashlib
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load, save_file
from .predictive import (SPEC as BASE_SPEC, Predictor, PatchEncoder, FrozenEncoder,
    _read, _json, _write, amplitudes, normalize, validate_parts, split_training,
    make_mask, objective, ema_update, _diagnostics, synthetic)

PUBLIC_BASELINE_SHA256='be5cbf9a740fa73e46bdb4ba3cdd25e2c41b5c47d8a629d906658816f7207f7c'

SPEC={'schema':'predictive_transfer_v1','base':BASE_SPEC,'epochs':5,'batch_size':8,'seed':42,
      'arms':{'baseline':{'record_variance':0.,'record_covariance':0.},
              'candidate':{'record_variance':1.,'record_covariance':.1}},
      'record_regularizer':'unmasked pooled32 embeddings; mean relu(1-sqrt(population_variance+1e-4)); squared offdiagonal unbiased covariance sum/32',
      'scope':'fixed research comparison inspired by VICReg, not reproduction; no confirmation or occupancy claim',
      'promotionAuthorized':False}


def record_penalties(embeddings):
    if embeddings.ndim!=2 or embeddings.shape[1]!=32 or len(embeddings)<2:
        raise ValueError('Record penalty requires at least two32-dimensional embeddings')
    centered=embeddings-embeddings.mean(0)
    variance=F.relu(1.-torch.sqrt(centered.square().mean(0)+1e-4)).mean()
    covariance=centered.T@centered/(len(embeddings)-1)
    off_diagonal=covariance-torch.diag_embed(covariance.diagonal())
    return variance,off_diagonal.square().sum()/32


def transfer_objective(student,teacher,x,mask,arm):
    if arm not in SPEC['arms']:raise ValueError('Unknown fixed arm')
    loss,diagnostic,target=objective(student,teacher,x,mask)
    if arm=='candidate':
        variance,covariance=record_penalties(student.encoder(x).mean(1))
        loss=loss+variance+.1*covariance
        diagnostic=dict(diagnostic,record_variance=variance.detach(),record_covariance=covariance.detach())
    return loss,diagnostic,target


def _state_hash(state):
    result=hashlib.sha256()
    for key,value in sorted(state.items()):
        result.update(key.encode());result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


class TransferEncoder(FrozenEncoder):
    """Versioned frozen online encoder; inherited bounded encode and adapter API."""
    def __init__(self,directory):
        root=Path(directory);status=_json(_read(root/'status.json',65536));report=_json(_read(root/'metadata.json',65536))
        if (not isinstance(status,dict) or not isinstance(report,dict) or status.get('status')!='complete'
                or report.get('status')!='complete' or status.get('promotionAuthorized') is not False
                or report.get('promotionAuthorized') is not False or report.get('spec')!=SPEC
                or report.get('arm') not in ('baseline','candidate','untrained')):
            raise ValueError('Invalid complete transfer artifact')
        data=_read(root/'model.safetensors',2*1024**2)
        if hashlib.sha256(data).hexdigest()!=report.get('model_sha256'):raise ValueError('Transfer model hash mismatch')
        state=load(data);self.encoder=PatchEncoder().eval().requires_grad_(False);expected=self.encoder.state_dict()
        if set(state)!=set(expected)|{'normalization.mean','normalization.scale'}:raise ValueError('Invalid transfer tensor keys')
        for key,value in expected.items():
            if state[key].shape!=value.shape or state[key].dtype!=value.dtype or not torch.isfinite(state[key]).all():raise ValueError('Invalid transfer tensor')
        self.encoder.load_state_dict({k:state[k] for k in expected})
        self.mean=state['normalization.mean'].numpy();self.scale=state['normalization.scale'].numpy()
        if (self.mean.shape!=(52,) or self.scale.shape!=(52,) or self.mean.dtype!=np.float64 or self.scale.dtype!=np.float64
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or not (self.scale>0).all()):
            raise ValueError('Invalid train normalization')


def _frozen(encoder,mean,scale):
    bundle=TransferEncoder.__new__(TransferEncoder);bundle.encoder=encoder.eval().requires_grad_(False)
    bundle.mean=mean;bundle.scale=scale
    return bundle


def _save_encoder(root,arm,encoder,mean,scale,source_hashes,lineage):
    root.mkdir();_write(root/'status.json',{'status':'running','promotionAuthorized':False})
    state={k:v.detach().cpu().contiguous() for k,v in encoder.state_dict().items()}
    state.update({'normalization.mean':torch.from_numpy(mean),'normalization.scale':torch.from_numpy(scale)})
    save_file(state,str(root/'model.safetensors'))
    replay=load(_read(root/'model.safetensors',2*1024**2))
    if any(not torch.equal(state[k],replay[k]) for k in state):raise ValueError('Export changed encoder')
    metadata={'status':'complete','promotionAuthorized':False,'spec':SPEC,'arm':arm,
              'model_sha256':hashlib.sha256(_read(root/'model.safetensors',2*1024**2)).hexdigest(),
              'source_sha256':source_hashes,'lineage':lineage}
    _write(root/'metadata.json',metadata)
    # Remains running until every campaign gate/evaluation and source check finishes.
    return metadata


def statistics(values):
    x=amplitudes(values).astype(np.float64)
    return np.concatenate((x.mean(-1),x.std(-1),np.abs(np.diff(x,axis=-1)).mean(-1)),axis=1)


def run(parts,lineage,output,max_seconds=240,baseline_artifact=None):
    if type(max_seconds) not in (int,float) or not math.isfinite(max_seconds) or not 1<=max_seconds<=3600:
        raise ValueError('Invalid time budget')
    validate_parts(parts)
    if not isinstance(lineage,dict) or lineage.get('evidence') not in ('SYNTHETIC','PUBLIC_TRAIN_ONLY'):
        raise ValueError('Explicit original-training or synthetic lineage required')
    import json
    if len(json.dumps(lineage,allow_nan=False))>16384:raise ValueError('Oversized lineage')
    for part in parts.values():
        y=np.asarray(part['y'])
        if y.shape!=(len(part['x']),) or y.dtype.kind not in 'iu' or set(y)!=set(range(7)):
            raise ValueError('Seven numeric activity classes required for frozen probe comparison')
    reference=None;reference_hash=None
    if baseline_artifact is not None:
        reference_bytes=_read(baseline_artifact,2*1024**2);reference_hash=hashlib.sha256(reference_bytes).hexdigest()
        if lineage['evidence']=='PUBLIC_TRAIN_ONLY' and reference_hash!=PUBLIC_BASELINE_SHA256:
            raise ValueError('Public original baseline artifact hash mismatch')
        reference=load(reference_bytes)
    elif lineage['evidence']=='PUBLIC_TRAIN_ONLY':
        raise ValueError('Original public baseline artifact is required')
    from .predictive_probe import run_probes
    names=('predictive_transfer.py','predictive.py','predictive_probe.py','espfi.py')
    source_hashes={n:hashlib.sha256(_read(Path(__file__).with_name(n),1024**2)).hexdigest() for n in names}
    root=Path(output);root.mkdir(parents=True,exist_ok=False);_write(root/'status.json',{'status':'running','promotionAuthorized':False})
    _write(root/'contract.json',SPEC)
    started=time.monotonic()
    def budget():
        if time.monotonic()-started>max_seconds:raise TimeoutError('Campaign budget exceeded between operations; no preemption guarantee')
    try:
        torch.manual_seed(42);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        x=parts['train']['x'].astype(np.float64);logged=np.log(np.maximum(x,1e-6))
        mean=logged.mean((0,2));scale=np.maximum(logged.std((0,2)),1e-6);training=normalize(x,mean,scale)
        initial=Predictor();initial_hash=_state_hash(initial.state_dict())
        trained={};histories={};traces={}
        for arm in ('baseline','candidate'):
            student=copy.deepcopy(initial);teacher=copy.deepcopy(student.encoder).eval().requires_grad_(False)
            optimizer=torch.optim.AdamW(student.parameters(),lr=1e-3,weight_decay=1e-4)
            generator=torch.Generator().manual_seed(42);trace=hashlib.sha256();history=[]
            for epoch in range(5):
                order=torch.randperm(len(training),generator=generator);losses=[]
                for offset in range(0,len(order),8):
                    budget();ids=order[offset:offset+8]
                    if len(ids)<2:continue
                    mask=make_mask(len(ids),generator);trace.update(ids.numpy().tobytes());trace.update(mask.numpy().tobytes())
                    optimizer.zero_grad(set_to_none=True)
                    loss,_,_=transfer_objective(student,teacher,training[ids],mask,arm)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite representation loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1.);optimizer.step();ema_update(teacher,student)
                    losses.append(float(loss.detach()))
                history.append({'epoch':epoch+1,'objective':float(np.mean(losses))})
            histories[arm]=history;traces[arm]=trace.hexdigest();trained[arm]=student.encoder.eval()
            if arm=='baseline' and reference is not None:
                actual=dict(student.encoder.state_dict(),**{'normalization.mean':torch.from_numpy(mean),'normalization.scale':torch.from_numpy(scale)})
                if set(actual)!=set(reference) or any(not torch.equal(actual[k],reference[k]) for k in actual):
                    raise ValueError('Original baseline state did not reproduce exactly')
        if len(set(traces.values()))!=1:raise ValueError('Arm mask or sample order differs')
        # Both fixed arms finish before any heldout representation is evaluated.
        _write(root/'representation_frozen.json',{'spec':SPEC,'initial_state_sha256':initial_hash,
            'mask_order_sha256':traces,'history':histories,'source_sha256':source_hashes})
        trained['untrained']=copy.deepcopy(initial.encoder).eval()
        frozen={name:_frozen(encoder,mean,scale) for name,encoder in trained.items()}
        metadata={name:_save_encoder(root/name,name,bundle.encoder,mean,scale,source_hashes,lineage) for name,bundle in frozen.items()}
        embeddings={};diagnostics={};timings={}
        for name,bundle in frozen.items():
            budget();embeddings[name]={'train':bundle.encode(parts['train']['x']),'dev':bundle.encode(parts['heldout']['x'])}
            # Validate exported state produces the same heldout embeddings before completion.
            replay=load(_read(root/name/'model.safetensors',2*1024**2));restored=PatchEncoder().eval().requires_grad_(False)
            restored.load_state_dict({k:replay[k] for k in restored.state_dict()})
            replay_bundle=_frozen(restored,replay['normalization.mean'].numpy(),replay['normalization.scale'].numpy())
            if not np.array_equal(replay_bundle.encode(parts['heldout']['x']),embeddings[name]['dev']):raise ValueError('Reload changes embeddings')
            with torch.no_grad():
                tokens=torch.cat([bundle.encoder(normalize(parts['heldout']['x'][i:i+8],mean,scale)) for i in range(0,len(parts['heldout']['x']),8)])
            diagnostics[name]=_diagnostics(tokens)
        # Same input and alternating order in each paired timing round.
        for i in range(10):
            j=i%len(parts['heldout']['x'])
            for bundle in frozen.values():bundle.encode(parts['heldout']['x'][j:j+1])
        timings={name:[] for name in frozen}
        for round_id in range(3):
            elapsed={name:[] for name in frozen}
            for i in range(100):
                budget();j=i%len(parts['heldout']['x']);sample=parts['heldout']['x'][j:j+1]
                calls=list(frozen.items())
                if i%2:calls.reverse()
                for name,bundle in calls:
                    before=time.perf_counter_ns();bundle.encode(sample)
                    elapsed[name].append((time.perf_counter_ns()-before)/1e6)
            for name,values in elapsed.items():
                timings[name].append({'round':round_id+1,'p50_ms':float(np.percentile(values,50)),
                                      'p95_ms':float(np.percentile(values,95))})
        embeddings['statistics']={key:statistics(parts[split]['x']) for key,split in [('train','train'),('dev','heldout')]}
        budget();probes=run_probes(embeddings,parts['train']['y'],parts['heldout']['y']);budget()
        checks={'probe_gate':probes['gate']['passed'],
                'candidate_effective_rank_at_least2':diagnostics['candidate']['recording_effective_rank']>=2.,
                'p95_within20percent_all_rounds':all(c['p95_ms']<=1.2*b['p95_ms'] for c,b in zip(timings['candidate'],timings['baseline']))}
        report={'status':'complete','promotionAuthorized':False,'spec':SPEC,'lineage':lineage,'evidence':lineage['evidence'],
                'initial_state_sha256':initial_hash,'mask_order_sha256':traces,'histories':histories,
                'original_baseline_sha256':reference_hash,'original_baseline_exact':True if reference is not None else None,
                'models':metadata,'collapse':diagnostics,'inference_latency':timings,'probes':probes,'checks':checks,
                'research_gate_passed':all(checks.values()),'embedding_shapes':{name:{k:list(v.shape) for k,v in rows.items()} for name,rows in embeddings.items()},
                'source_sha256':source_hashes,'torch_version':torch.__version__,'numpy_version':np.__version__,
                'elapsed_seconds':time.monotonic()-started,'scope':'Fixed two-arm adaptive research on reused original training groups. No original validation/test payload, no heldout updates or selection. Encoder timings3x100 alternating paired batch1 include checks/normalization/encode, exclude acquisition/IO. No occupancy or confirmation claim.'}
        if any(hashlib.sha256(_read(Path(__file__).with_name(n),1024**2)).hexdigest()!=value for n,value in source_hashes.items()):raise ValueError('Source changed during campaign')
        budget();_write(root/'metrics.json',report)
        for name in frozen:_write(root/name/'status.json',{'status':'complete','promotionAuthorized':False})
        _write(root/'status.json',{'status':'complete','promotionAuthorized':False})
        return report
    except Exception:
        for name in ('baseline','candidate','untrained'):
            if (root/name).is_dir():_write(root/name/'status.json',{'status':'failed','promotionAuthorized':False})
        _write(root/'status.json',{'status':'failed','promotionAuthorized':False});raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--synthetic',action='store_true');group.add_argument('--manifest');parser.add_argument('--output',required=True)
    parser.add_argument('--max-seconds',type=int,default=240);parser.add_argument('--baseline-artifact');args=parser.parse_args()
    if args.synthetic:
        parts,lineage=synthetic()
        for part in parts.values():part['y']=np.arange(len(part['x']),dtype=np.int64)%7
    else:
        if not args.baseline_artifact:parser.error('--baseline-artifact is required for public comparison')
        from .espfi import _load_partitions
        admitted,meta=_load_partitions(args.manifest,('train',));part=admitted['train'];parts=split_training(part['x'],part['identity'])
        ids=part['identity'];fit=(ids[:,0]==1)&np.isin(ids[:,1],[1,2,3,4]);held=(ids[:,0]==2)&np.isin(ids[:,1],[5,6])
        parts['train']['y']=part['y'][fit];parts['heldout']['y']=part['y'][held]
        lineage={'evidence':'PUBLIC_TRAIN_ONLY','manifest_sha256':meta['manifest_sha256'],'loaded_partitions':['train'],
                 'source_revision':meta['source_revision'],'inner_fit':'room1subjects1..4','inner_heldout':'room2subjects5,6'}
    import json
    print(json.dumps(run(parts,lineage,args.output,args.max_seconds,args.baseline_artifact),indent=2))

if __name__=='__main__':main()
