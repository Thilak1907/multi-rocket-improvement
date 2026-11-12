"""Correctness-first review: byte routing, isolated ablation, and non-inferiority.

All new MultiRocket paths use corrected first-difference kernels. Historical
results and implementation snapshots remain separate. No confirmation labels
enter the choice of bundle. The predeclared margin is -0.3 percentage points.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time

from experiments.compare_optimizations import atomic_json,digest,seed_for,load_data
from experiments.search_combinations import CONFIGS as ORIGINAL, SOURCES as ORIGINAL_SOURCES

ROOT=Path(__file__).resolve().parents[1]
FULL=dict(num_features=30000,classifier='ridge',memory_efficient=True,
          optimized_transform=True,fused_prediction=True,first_difference_fix=True,
          prediction_batch_size=256)
PLAIN=dict(num_features=30000,classifier='ridge',memory_efficient=False,
           optimized_transform=False,fused_prediction=False,first_difference_fix=True)
ABLATIONS={
 'baseline30':dict(PLAIN),
 'feature50':{**PLAIN,'num_features':50000},
 'blocked_only':{**PLAIN,'classifier':'fast_ridge'},
 'buffered_only':{**PLAIN,'optimized_transform':True},
 'inplace_only':{**PLAIN,'inplace_processing':True},
 'batching_only':{**PLAIN,'chunked_prediction':True,'prediction_batch_size':256},
 'folding_only':{**PLAIN,'fused_prediction':True},
 'routing_only':{**PLAIN,'classifier':'auto_ridge'},
 'full_auto':{**FULL,'classifier':'auto_ridge'},
 'full_no_fold':{**FULL,'classifier':'auto_ridge','fused_prediction':False},
}
GRAPH={**{k:dict(v) for k,v in ORIGINAL.items()},'fast30':{**FULL,'classifier':'fast_ridge'}}
FINALISTS=['baseline50','fast30','fused30','auto30','fused10','hybrid20h32']
ACCURACY=['baseline50','fast30','auto30']
SOURCES=list(dict.fromkeys(ORIGINAL_SOURCES+['multirocket/corrected.py','experiments/review_revision.py']))


def initialize(args):
    import numpy as np,pandas as pd,numba,scipy,sklearn,torch
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    file=out/'protocol.json'
    sources={p:digest(ROOT/p) for p in SOURCES}
    old=json.loads((ROOT/'output/combination_search/20261111/protocol.json').read_text())
    if file.exists():
        protocol=json.loads(file.read_text())
        if protocol['sources']!=sources: raise ValueError('Source changed; use a new output directory')
        if protocol['max_rss_gib']!=args.max_rss_gib: raise ValueError('Memory guard changed')
    else:
        # Workload-only representative selection, frozen before any ablation.
        sizes=pd.read_csv(ROOT/'data/classification_datasets_109.csv').set_index('Name')
        ordered=sorted(old['split']['selection'],key=lambda d:float(sizes.loc[d,'Train '])*float(sizes.loc[d,'Length']))
        representative=list(dict.fromkeys([ordered[i] for i in np.linspace(0,len(ordered)-1,12).astype(int)]))
        for d in ('ElectricDevices','Crop'):
            if d not in representative: representative[-2 if d=='ElectricDevices' else -3]=d
        protocol=dict(sources=sources,split=old['split'],seed=old['seed'],datapath=old['datapath'],
          data_hashes=old['data_hashes'],index_hashes={d:digest(ROOT/f'data/indices109/{d}_INDICES_TRAIN.txt')
                        for d in old['split']['selection']+old['split']['heldout']},
          threads=4,blas_threads=1,max_rss_gib=args.max_rss_gib,timing_repeats=3,resamples=[1,2,3],
          first_difference_fix=True,ridge_feature_limit_bytes=693*1024**2,
          ridge_rule='blocked if n_train * actual_feature_columns * sizeof(float32) >= 693 MiB; sklearn otherwise',
          graph_configs=GRAPH,ablation_configs=ABLATIONS,finalists=FINALISTS,accuracy_finalists=ACCURACY,
          ablation_datasets=representative,ablation_train_cap=512,ablation_test_cap=512,
          ablation_rule='Keep the full bundle unless dropping folding preserves all predictions and alpha, and either improves time/RSS geometric cost by >=1%, or saves >=5% model size with time <=1.01x and RSS <=1.02x. Other components are measured individually.',
          noninferiority_margin_pp=-.3,noninferiority_alpha=.05,
          accuracy_primary='55 per-dataset paired accuracy differences averaged over resamples 1,2,3; training-resample repetitions are not independent dataset observations',
          noninferiority_rule='Predeclared one-sided paired t-test at .05 against -0.3 pp, plus two-sided dataset-bootstrap 95% lower bound above -0.3 pp',
          adoption_rule='Adopt auto30 if non-inferiority gate passes, resample-1 runtime >=3% faster than corrected fast30, paired median RSS <=1.20x fast30, and ElectricDevices completes below the 9 GiB guard. Otherwise keep the previous fixed backend; legacy feature mode remains explicit.',
          audit_datasets=['BME','CinCECGTorso','FordA','SonyAIBORobotSurface1','WormsTwoClass'],
          platform=platform.platform(),python=sys.version,
          versions=dict(numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,numba=numba.__version__,torch=torch.__version__))
        atomic_json(file,protocol)
        for p in SOURCES:
            target=out/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/p).read_bytes())
    for p,sha in protocol['data_hashes'].items():
        if digest(Path(protocol['datapath'])/p)!=sha: raise ValueError('Data changed: '+p)
    for d,sha in protocol['index_hashes'].items():
        if digest(ROOT/f'data/indices109/{d}_INDICES_TRAIN.txt')!=sha: raise ValueError('Indices changed: '+d)
    return protocol


def capped_indices(y,cap,rng):
    import numpy as np
    if len(y)<=cap: return np.arange(len(y))
    labels,counts=np.unique(y,return_counts=True)
    quota=np.maximum(1,np.floor(counts*cap/len(y)).astype(int))
    while quota.sum()>cap:
        i=int(np.argmax(quota));quota[i]-=1
    while quota.sum()<cap:
        room=counts-quota;i=int(np.argmax(room));quota[i]+=1
    return np.sort(np.concatenate([rng.permutation(np.flatnonzero(y==label))[:q]
                                  for label,q in zip(labels,quota)]))


def dataset(dataset,phase,resample,protocol):
    import numpy as np
    X,y,Xt,yt=load_data(dataset,protocol)
    rng=np.random.default_rng(seed_for(dataset,protocol['seed']))
    if phase=='ablation':
        keep,val=[],[]
        for label in np.unique(y):
            idx=rng.permutation(np.flatnonzero(y==label)); n=min(len(idx)-1,max(1,int(round(.2*len(idx)))))
            val.extend(idx[:n]);keep.extend(idx[n:])
        keep,val=np.sort(keep),np.sort(val)
        keep=keep[capped_indices(y[keep],protocol['ablation_train_cap'],rng)]
        val=val[capped_indices(y[val],protocol['ablation_test_cap'],rng)]
        result=(X[keep],y[keep],X[val],y[val])
    else:
        allX,ally=np.concatenate((X,Xt)),np.concatenate((y,yt))
        keep=np.loadtxt(ROOT/f'data/indices109/{dataset}_INDICES_TRAIN.txt',skiprows=resample,max_rows=1).astype(np.int64)
        val=np.setdiff1d(np.arange(len(allX)),keep)
        if len(np.unique(keep))!=len(X): raise ValueError('Invalid indices for '+dataset)
        result=(allX[keep],ally[keep],allX[val],ally[val])
    return (*result,dict(train_indices_sha256=hashlib.sha256(keep.tobytes()).hexdigest(),
                        test_indices_sha256=hashlib.sha256(val.tobytes()).hexdigest()))


def configs(out,protocol):
    values={k:dict(v) for k,v in protocol['graph_configs'].items()}
    choice=json.loads((out/'bundle_choice.json').read_text())
    values['auto30']=choice['configuration']
    return values


def freeze_bundle(out,protocol):
    import pandas as pd,numpy as np
    rows=[];hashes={}
    for d in protocol['ablation_datasets']:
        for v in ABLATIONS:
            for r in range(3):
                file=out/'ablation'/f'{d}__{v}__s1__r{r}.json'
                value=json.loads(file.read_text())
                if value['status']!='ok': raise ValueError('Incomplete ablation '+str(file))
                rows.append(value);hashes[str(file.relative_to(out))]=digest(file)
    raw=pd.DataFrame(rows)
    m=raw.groupby(['dataset','variant'])[['total_s','peak_rss_bytes','model_bytes']].median()
    full=m.xs('full_auto',level='variant'); lean=m.xs('full_no_fold',level='variant')
    tr=float(lean.total_s.sum()/full.total_s.sum()); rr=float((lean.peak_rss_bytes/full.peak_rss_bytes).median())
    storage=float((lean.model_bytes/full.model_bytes).median())
    a=raw[raw.variant=='full_auto'].set_index(['dataset','repeat'])
    b=raw[raw.variant=='full_no_fold'].set_index(['dataset','repeat']).loc[a.index]
    parity=bool((a.alpha==b.alpha).all() and (a.prediction_hash==b.prediction_hash).all())
    drop=parity and (np.sqrt(tr*rr)<=.99 or (storage<=.95 and tr<=1.01 and rr<=1.02))
    configuration={**FULL,'classifier':'auto_ridge','fused_prediction':not drop,
                   'ridge_feature_limit_bytes':protocol['ridge_feature_limit_bytes']}
    choice=dict(configuration=configuration,folding_removed=bool(drop),time_ratio_no_fold=tr,
                rss_ratio_no_fold=rr,model_ratio_no_fold=storage,exact_ablation_parity=parity,ablation_hashes=hashes)
    file=out/'bundle_choice.json'
    if file.exists() and json.loads(file.read_text())!=choice: raise ValueError('Frozen bundle choice changed')
    if not file.exists(): atomic_json(file,choice)
    return choice


def warmup():
    import numpy as np
    from multirocket.multirocket import MultiRocket
    from experiments.hydra_model import HydraRocket
    X=np.random.default_rng(1).normal(size=(8,32)); y=np.arange(8)%2
    for options in ({},{'optimized_transform':True,'memory_efficient':True},
                    {'pooling':('LSPV','MPV'),'classifier':'fast_ridge'}):
        m=MultiRocket(672,random_state=1,**options);m.fit(X,y,False);m.predict(X[:2])
    m=HydraRocket(groups=4);m.fit(X,y);m.predict(X[:2])


def worker(args):
    import numpy as np,numba,torch,resource
    from threadpoolctl import threadpool_limits
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
    from multirocket.multirocket import MultiRocket
    from experiments.hydra_model import HydraRocket
    out=Path(args.output);p=json.loads((out/'protocol.json').read_text())
    numba.set_num_threads(p['threads']);torch.set_num_threads(p['threads']);torch.set_num_interop_threads(1)
    with threadpool_limits(limits=1,user_api='blas'):
        warmup();X,y,Xt,yt,hashes=dataset(args.dataset,args.worker,args.resample,p)
        if args.worker=='ablation': options=dict(p['ablation_configs'][args.variant])
        elif args.worker=='audit': options={**FULL,'classifier':'fast_ridge','first_difference_fix':False}
        else: options=dict(configs(out,p)[args.variant])
        seed=seed_for(args.dataset,p['seed'])
        if 'hydra_groups' in options:
            m=HydraRocket(budget=options['num_features'],groups=options['hydra_groups'],seed=seed,first_difference_fix=True)
        else:
            options['random_state']=seed;options['first_difference_fix']=args.worker!='audit'
            m=MultiRocket(**options)
        t=time.perf_counter();m.fit(X,y,predict_on_train=False);fit_s=time.perf_counter()-t
        t=time.perf_counter();pred=m.predict(Xt);predict_s=time.perf_counter()-t
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        backend=m.selected_ridge_backend_ if isinstance(m,MultiRocket) else 'ridge'
        alpha=float(m.classifier[-1].alpha_) if isinstance(m,MultiRocket) else m.alpha_
        row=dict(dataset=args.dataset,variant=args.variant,resample=args.resample,repeat=args.repeat,status='ok',
          seed=seed,first_difference_fix=args.worker!='audit',**hashes,n_train=len(X),n_test=len(Xt),length=X.shape[1],
          accuracy=float(accuracy_score(yt,pred)),balanced_accuracy=float(balanced_accuracy_score(yt,pred)),
          macro_f1=float(f1_score(yt,pred,average='macro',zero_division=0)),
          fit_s=fit_s,predict_s=predict_s,total_s=fit_s+predict_s,
          classifier_fit_s=m.train_duration,classifier_predict_s=m.test_duration,
          train_transform_s=m.apply_kernel_on_train_duration,test_transform_s=m.apply_kernel_on_test_duration,
          peak_rss_bytes=rss,features=m.n_features_out_,alpha=alpha,selected_backend=backend,
          train_feature_bytes=m.train_feature_bytes_,max_prediction_feature_bytes=m.max_prediction_feature_bytes_,
          model_bytes=len(pickle.dumps(m,protocol=5)),prediction_hash=hashlib.sha256(pred.tobytes()).hexdigest())
        if args.worker=='audit':
            previous=ROOT/'output/combination_search/20261111_fast30/confirmation'/f'{args.dataset}__fast30__0.json'
            old=json.loads(previous.read_text())
            row['legacy_reference_prediction_equal']=old['prediction_hash']==row['prediction_hash']
            row['legacy_reference_alpha_equal']=old['alpha']==row['alpha']
        stem=f'{args.dataset}__{args.variant}__s{args.resample}__r{args.repeat}'
        target=out/args.worker/stem
        np.save(target.with_suffix('.npy'),pred);atomic_json(target.with_suffix('.json'),row)


def job(args,phase,dataset,variant,resample=1,repeat=0):
    import psutil
    out=Path(args.output);folder=out/phase;folder.mkdir(exist_ok=True)
    stem=f'{dataset}__{variant}__s{resample}__r{repeat}';target=folder/f'{stem}.json'
    if target.exists(): return json.loads(target.read_text())['status']=='ok'
    command=[sys.executable,'-m','experiments.review_revision','--worker',phase,'--output',str(out),
             '--dataset',dataset,'--variant',variant,'--resample',str(resample),'--repeat',str(repeat)]
    env=dict(os.environ,VECLIB_MAXIMUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
    with (folder/f'{stem}.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        process=psutil.Process(child.pid);peak=0;status='failed';start=time.monotonic()
        while child.poll() is None:
            try:
                peak=max(peak,process.memory_info().rss)
                if peak>args.max_rss_gib*1024**3: status='rss_limit';child.kill()
                elif time.monotonic()-start>1200: status='timeout';child.kill()
            except psutil.NoSuchProcess: pass
            time.sleep(.05)
        if child.returncode!=0:
            atomic_json(target,dict(dataset=dataset,variant=variant,resample=resample,repeat=repeat,
                                    status=status,peak_rss_bytes=peak,exitcode=child.returncode))
            print(f'  {status}: {stem}',flush=True);return False
    if not target.exists(): raise RuntimeError('Missing worker result: '+stem)
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='output/review_revision/corrected_v1')
    parser.add_argument('--stage',choices=('all','ablation','confirmation','accuracy','stress','audit'),default='all')
    parser.add_argument('--worker',choices=('ablation','confirmation','accuracy','stress','audit'))
    parser.add_argument('--dataset');parser.add_argument('--variant')
    parser.add_argument('--resample',type=int,default=1);parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--max-rss-gib',type=float,default=9)
    args=parser.parse_args()
    if args.worker: worker(args);return
    p=initialize(args);out=Path(args.output)
    if args.stage in ('all','ablation'):
        for i,d in enumerate(p['ablation_datasets']):
            for r in range(3):
                variants=list(ABLATIONS);shift=(i+r)%len(variants)
                for v in variants[shift:]+variants[:shift]:
                    print(f'ablation {i+1}/{len(p["ablation_datasets"])} {d} {v} repeat={r}',flush=True)
                    if not job(args,'ablation',d,v,1,r): raise RuntimeError('Ablation incomplete')
        print('Frozen bundle: '+str({k:v for k,v in freeze_bundle(out,p).items() if k!='ablation_hashes'}),flush=True)
    if args.stage in ('all','confirmation','accuracy','stress'):
        freeze_bundle(out,p)
    if args.stage in ('all','confirmation'):
        for i,d in enumerate(p['split']['heldout']):
            for r in range(3):
                variants=list(dict.fromkeys(list(GRAPH)+['auto30'])) if r==0 else FINALISTS
                shift=(i+r)%len(variants)
                for v in variants[shift:]+variants[:shift]:
                    print(f'confirmation {i+1}/55 {d} {v} repeat={r}',flush=True)
                    job(args,'confirmation',d,v,1,r)
    if args.stage in ('all','accuracy'):
        for s in (2,3):
            for i,d in enumerate(p['split']['heldout']):
                variants=ACCURACY;shift=(i+s)%len(variants)
                for v in variants[shift:]+variants[:shift]:
                    print(f'accuracy resample={s} {i+1}/55 {d} {v}',flush=True)
                    job(args,'accuracy',d,v,s,0)
    if args.stage in ('all','stress'):
        for v in FINALISTS:
            print('stress ElectricDevices '+v,flush=True);job(args,'stress','ElectricDevices',v,1,0)
    if args.stage in ('all','audit'):
        for d in p['audit_datasets']:
            print('legacy audit '+d,flush=True);job(args,'audit',d,'legacy_fast30',1,0)


if __name__=='__main__': main()
