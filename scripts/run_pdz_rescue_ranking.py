"""Frozen sequential method comparison, nested calibration and domain transfer."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy.special import ndtr
from pdz2018_data import ROOT, DOMAINS, RT, dump, hashes, load_development
from pdz_background_model import pair_features, OnlinePredictor, site
from pdz_rescue_ranking import Model, Search, METHODS, THRESHOLD, public_task, features

OUT=ROOT/'experiments/pdz2018_rescue_ranking_v1'
KINDS=['ridge','rank']


def freeze():
    assert not (OUT/'freeze.json').exists()
    OUT.mkdir(exist_ok=True)
    protocol=dict(version=OUT.name,created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Exploratory development after prior result inspection; not independent confirmation',
        tasks=207,domains=list(DOMAINS),sealed_unread=['shank3','zo1'],budget=32,initial_wt=8,batch=4,
        endpoint='exp(-G/RT)>=0.5 WT; point energy only; all tasks retained',
        access='Historical canonical development double genotypes allowed to all learned arms; WT uses public singles',
        primary='Nine site folds exclude every genotype touching outer damage site at either endpoint across all domains',
        transfer='Three leave-one-development-domain-out folds exclude all genotypes from held domain; same207 tasks and pools',
        inner='Three inner site groups modulo3; exclude both endpoints touching inner sites plus entire outer exclusion. Calibration pools exclude outer sites/domains. No outer labels used for fit/calibration/parameter selection',
        methods=METHODS,stages={'1':'conservative_lcb vs ridge_full_lcb; change budget only',
          '2':'conservative_pi vs conservative_lcb; change acquisition only',
          '3':'rank_pi vs conservative_pi; change model objective, including its identically fitted calibration',
          '4':'ensemble_pi vs conservative_pi and rank_pi; arithmetic average of calibrated probabilities'},
        conservative='After initial8, three unqueried WT anchors plus one expert; partial final batch all WT',
        ridge='Same structural features and alpha10 as previous ridge; train-only scaling',
        ranking='Same features; 0.5 MSE + 0.5 pairwise logistic loss within damaging backgrounds + (10/n)*L2. 32 seeded pairs/background; seed20261008; no sweep; max1000 L-BFGS iterations',
        calibration='Inner OOF GP errors after8/16/24 WT queries; equal snapshot weight, equal candidate weight within snapshot; fit mean bias and RMS normalized scale. Gaussian probability with latent variance plus .09. Transfer this fixed calibration to full outer refit; report Brier and limits',
        online='Existing structural RBF GP and noise .09; update only committed queried energies; full legal pool; deterministic tie IDs',
        ensemble='Equal average ridge/rank probabilities, no learned gate and no outcome-selected expert routing',
        metrics=['Success@8/16/24/32','first_success_completed_batch_cost; failures=cap+4','paired gains/losses','domain/site results','post-initial8 MAE and Brier','site-cluster paired bootstrap'],
        selection='Among four new methods, maximize site Success@32, tie lower penalized first-batch cost, then listed method order. No parameter tuning',
        agnes_gate={'net_site_vs_strongest_reference_at_least':3,'gained_sites_vs_WT_at_least':2,
                    'net_transfer_vs_strongest_reference_at_least':1,'worst_transfer_domain_vs_WT_at_least':-1},
        references=METHODS[:3],agnes='Only if selected method passes all frozen gates and verification: new32-task same original identity cohort with full legal menu, strongest fixed strategy and WT, same budget; no sealed test',
        confidence='Bootstrap describes grouped development differences, not biological replication or independent final evidence',
        source='https://www.nature.com/articles/s41467-026-76264-2; motivation only, this is a smaller custom linear model')
    dump(OUT/'protocol.json',protocol)
    preserved={}
    for folder in ['gb1_final_holdout_v1','pdz2018_point_dev_v1','pdz2018_uncertainty_v1','pdz2018_background_v1']:
        preserved.update(hashes([p for p in (ROOT/'experiments'/folder).rglob('*') if p.is_file()]))
    dump(OUT/'preserved_hashes.json',preserved)
    paths=[ROOT/'scripts'/n for n in ['pdz_rescue_ranking.py','run_pdz_rescue_ranking.py','verify_pdz_rescue_ranking.py',
        'pdz_background_model.py','pdz2018_data.py','pdz2018_policy.py']]
    paths += [OUT/'protocol.json',OUT/'preserved_hashes.json',ROOT/'experiments/pdz2018_background_v1/historical_training_genotypes.csv.gz',
        ROOT/'data/pdz2018/source/Salinas_Ranganathan_data.mat',ROOT/'data/public_reference/residue_volumes.json',
        ROOT/'experiments/dataset_feasibility_v1/pdz2018_calibration_QC.csv']
    paths += sorted((ROOT/'data/pdz2018/public').glob('*_geometry.npz'))
    dump(OUT/'freeze.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes(paths)))
    print('Protocol and input hashes frozen',flush=True)


def check_hashes():
    for name in ['freeze.json','preserved_hashes.json']:
        obj=json.loads((OUT/name).read_text())
        for p,h in obj.get('hashes',obj).items():
            assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p


def outer_mask(records,scheme,held):
    if scheme=='site':return np.array([held not in (r['left_site'],r['right_site']) for r in records])
    return np.array([r['domain']!=held for r in records])


def calibrate(records,x,tasks,scheme,held,mask):
    collected={k:[] for k in KINDS}; audits=[]
    for inner in range(3):
        inner_sites={s for s in range(9) if s%3==inner}
        selected=mask & np.array([not (inner_sites & {r['left_site'],r['right_site']}) for r in records])
        train=[r for r,m in zip(records,selected) if m]
        models={k:Model(train,x[selected],k=='rank') for k in KINDS}
        inner_tasks=0
        for t in tasks.values():
            ds=site(t['damage'])
            if ds not in inner_sites or (scheme=='site' and ds==held) or (scheme=='domain' and t['domain']==held):continue
            rows=[r for r in t['rows'] if scheme!='site' or site(r['id'])!=held]
            if len(rows)<=8:continue
            inner_tasks+=1
            private={**t,'rows':rows};public=public_task(private);xf=features(public)
            order=np.lexsort((np.array([r['id'] for r in rows]),-np.array([r['WT_log_affinity'] for r in rows])))
            energy=np.array([r['deltaG_kcal'] for r in rows])
            for k,model in models.items():
                pred=OnlinePredictor(public,model.predict(xf),True)
                for n in [8,16,24]:
                    if len(rows)<=n:break
                    new=list(map(int,order[len(pred.seen):n]));pred.update(new,energy[new]-pred.additive[new])
                    mean,cov=pred.posterior();mu=pred.additive+mean
                    sd=np.sqrt(np.maximum(np.diag(cov),1e-12)+pred.noise)
                    unseen=order[n:]
                    for i in unseen:
                        collected[k].append((energy[i]-mu[i],sd[i],mu[i],energy[i]<=THRESHOLD,1/len(unseen)))
        audits.append(dict(inner=inner,sites=sorted(inner_sites),training_n=len(train),calibration_tasks=inner_tasks,
            training_key_hash=hashlib.sha256(json.dumps([(r['domain'],r['left'],r['right']) for r in train]).encode()).hexdigest(),
            fits={k:m.audit for k,m in models.items()}))
    calibration={}
    for k,values in collected.items():
        e,sd,mu,y,w=np.array(values).T;w=w/w.sum()
        bias=float(w@e);scale=float(np.sqrt(w@(((e-bias)/sd)**2)))
        scale=max(scale,1e-6)
        raw=ndtr((THRESHOLD-mu)/sd);cal=ndtr((THRESHOLD-mu-bias)/(sd*scale))
        calibration[k]=dict(bias=bias,scale=scale,n=len(e),raw_brier=float(w@((raw-y)**2)),
                            calibrated_brier_fit=float(w@((cal-y)**2)))
    return calibration,audits


def evaluate():
    check_hashes()
    assert not (OUT/'started.json').exists(),'Use a new run after interruption; never silently overwrite'
    dump(OUT/'started.json',dict(at_utc=datetime.now(timezone.utc).isoformat()))
    tasks=load_development();assert len(tasks)==207
    records=pd.read_csv(ROOT/'experiments/pdz2018_background_v1/historical_training_genotypes.csv.gz').to_dict('records')
    x=np.array([pair_features(r['left'],r['right'],r['gd'],r['gr'],r['distance'],True) for r in records])
    runs=[];metrics=[];audits=[]
    (OUT/'models').mkdir()
    with gzip.open(OUT/'trajectories.jsonl.gz','wt') as stream:
        for scheme,held in [('site',s) for s in range(9)]+[('domain',d) for d in DOMAINS]:
            fold=f'{scheme}_{held}';print('Fitting',fold,flush=True)
            mask=outer_mask(records,scheme,held);train=[r for r,m in zip(records,mask) if m]
            calibration,inner_audit=calibrate(records,x,tasks,scheme,held,mask)
            models={k:Model(train,x[mask],k=='rank') for k in KINDS}
            for k,m in models.items():m.save(OUT/'models'/f'{fold}_{k}.npz')
            dump(OUT/'models'/f'{fold}_calibration.json',calibration)
            audits.append(dict(fold=fold,training_n=len(train),inner=inner_audit,fits={k:m.audit for k,m in models.items()}))
            trainkeys={(r['domain'],r['left'],r['right']) for r in train}
            for tid,t in tasks.items():
                if (scheme=='site' and site(t['damage'])!=held) or (scheme=='domain' and t['domain']!=held):continue
                for r in t['rows']:
                    left,right=sorted([t['damage'],r['id']],key=site)
                    assert (t['domain'],left,right) not in trainkeys
                public=public_task(t);xf=features(public);bases={k:m.predict(xf) for k,m in models.items()}
                energy=np.array([r['deltaG_kcal'] for r in t['rows']]);ids=[r['id'] for r in t['rows']]
                for method in METHODS:
                    policy=Search(public,method,bases,calibration);chosen=[];batches=[];first=None
                    while len(chosen)<policy.wt.cap:
                        batch=policy.proposal();before=len(chosen)
                        assert len(set(batch))==len(batch) and not set(batch)&set(chosen)
                        # Labels are provided only after this entire batch is committed.
                        policy.update(batch,energy[batch]);chosen+=batch
                        batches.append([ids[i] for i in batch])
                        if first is None and np.min(energy[chosen])<=THRESHOLD:first=len(chosen)
                        if before==4 and method=='wt_rank':
                            unseen=[i for i in range(len(ids)) if i not in chosen]
                            for k,p in policy.models.items():
                                mean,cov=p.posterior();mu=p.additive+mean
                                from pdz_rescue_ranking import probability
                                prob=probability(p,calibration[k]);truth=energy<=THRESHOLD
                                metrics.append(dict(scheme=scheme,task_id=tid,model=k,MAE=float(np.mean(abs(mu[unseen]-energy[unseen]))),
                                    Brier=float(np.mean((prob[unseen]-truth[unseen])**2))))
                    success={f'success{k}':bool(np.min(energy[chosen[:k]])<=THRESHOLD) for k in [8,16,24,32]}
                    runs.append(dict(scheme=scheme,fold=fold,task_id=tid,domain=t['domain'],damage=t['damage'],damage_site=site(t['damage'])+1,
                        method=method,queries=len(chosen),**success,first_batch_cost=first,penalized_cost=first if first else policy.wt.cap+4,
                        best_affinity=float(np.exp(-min(energy[chosen])/RT)),reachable=bool(min(energy)<=THRESHOLD)))
                    stream.write(json.dumps(dict(scheme=scheme,fold=fold,task_id=tid,method=method,batches=batches))+'\n')
            stream.flush();dump(OUT/'fit_audits.json',audits)
            pd.DataFrame(runs).to_csv(OUT/'runs.csv',index=False)
            pd.DataFrame(metrics).to_csv(OUT/'prediction_metrics.csv',index=False)
            print('Completed',fold,'cumulative trajectories',len(runs),flush=True)
    check_hashes()
    dump(OUT/'execution.json',dict(status='completed',trajectories=len(runs),new_agnes_calls=0,sealed_domains_loaded=False))


if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='evaluate':evaluate()
    else:raise ValueError('freeze or evaluate')
