"""Frozen ESM residual-head comparison with conditional expert fusion."""
from datetime import datetime,timezone
import gzip
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from pdz2018_data import ROOT,DOMAINS,RT,dump,hashes,load_development
from pdz_background_model import site
from pdz_rescue_ranking import Model,THRESHOLD,public_task,features,probability
from pdz_esm_model import OUT,OLD,METHODS,FUSION,ESMFeatures,search,calibrate
from run_pdz_rescue_ranking import outer_mask
from prepare_pdz_esm import check as feature_check


def freeze():
    assert not (OUT/'freeze.json').exists()
    p=dict(version=OUT.name,scope='New exploratory development; same assay point labels; not independent validation',
        tasks=207,budget=32,initial_wt=8,batch=4,sealed_unread=['shank3','zo1'],
        features='Existing structural features plus384 frozen ESM2-35M full-construct projected features; same ridge alpha10 and GP; train-only scaling',
        control='Exact old structural model and nested calibration files from rescue_ranking_v1; verify trajectories exactly',
        splits='Nine site folds exclude both genotype endpoints across all domains; three leave-domain-out folds',
        calibration='Same3 inner site groups and WT8/16/24 calibration snapshots as previous round; outer exclusion applies to inner training and calibration; Gaussian error bias/scale',
        methods=METHODS,comparisons=[['esm_full_lcb','ridge_full_lcb'],['esm_pi','conservative_pi']],
        selection='Choose between esm_full_lcb and esm_pi by site success32, tie lower penalized first-batch cost, tie esm_pi. No model size/seed/alpha sweep',
        conditional_fusion='Evaluate predeclared equal-probability esm_ensemble_pi only if esm_pi gains at least1 task vs conservative_pi and loses no net tasks on site-held development. Same WT3+expert1 allocation; label-informed stage gate disclosed',
        fusion_selection='Promote fusion only if site success32 strictly greater than strongest single expert on the same conservative allocation; otherwise retain preselected single model',
        agnes_gate=dict(net_site_vs_best_original_at_least=3,net_transfer_vs_best_original_at_least=1,gained_sites_vs_WT_at_least=2,worst_transfer_domain_vs_WT_at_least=-1),
        metrics=['success8/16/24/32','paired gains/losses','first_completed_batch_cost;failures=cap+4','domain results','shared8 MAE/Brier','exploratory paired grouped bootstrap'],
        limits=['Complete PDB constructs used as reference sequence context; full experimental plasmid sequences not confirmed',
                'Pretrained natural-sequence exposure is allowed; supervised assay-label leakage prohibited',
                'Frozen35M model with128D fixed projection; negative results do not rule out all PLMs or larger models',
                'PDZ3 biological uncertainty unavailable; point-only; no sealed tests or wet-lab claims'])
    dump(OUT/'protocol.json',p)
    preserved={}
    for folder in ['pdz2018_background_v1','pdz2018_rescue_ranking_v1','pdz2018_point_dev_v1','pdz2018_uncertainty_v1','gb1_final_holdout_v1']:
        preserved.update(hashes([f for f in (ROOT/'experiments'/folder).rglob('*') if f.is_file()]))
    dump(OUT/'preserved_hashes.json',preserved)
    paths=[ROOT/'scripts'/n for n in ['pdz_esm_model.py','run_pdz_esm.py','verify_pdz_esm.py','report_pdz_esm.py',
           'prepare_pdz_esm.py','pdz_rescue_ranking.py','pdz_background_model.py','pdz2018_data.py','pdz2018_policy.py','run_pdz_rescue_ranking.py']]
    paths += [OUT/'protocol.json',OUT/'preserved_hashes.json',OUT/'feature_freeze.json',OUT/'feature_protocol.json',OUT/'sequence_mapping.json']
    dump(OUT/'freeze.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes(paths),boundary='fixed before new head fits/search outcomes; representation extraction is label-free'))


def check():
    feature_check()
    for name in ['freeze.json','preserved_hashes.json']:
        obj=json.loads((OUT/name).read_text())
        for p,h in obj.get('hashes',obj).items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    if (OUT/'embedding_manifest.json').exists():
        for r in json.loads((OUT/'embedding_manifest.json').read_text())['domains']:
            assert hashlib.sha256((OUT/f"{r['domain']}_embeddings.npz").read_bytes()).hexdigest()==r['sha256']


def simulate(tasks,store,models,calibration,scheme,fold,methods,stream):
    runs=[];metrics=[]
    for tid,t in tasks.items():
        public=public_task(t);base={'old':models['old'].predict(features(public)),'esm':models['esm'].predict(store.task(public))}
        energy=np.array([r['deltaG_kcal'] for r in t['rows']]);ids=[r['id'] for r in t['rows']]
        for method in methods:
            p=search(public,method,base,calibration);chosen=[];batches=[];first=None
            while len(chosen)<p.wt.cap:
                batch=p.proposal();before=len(chosen)
                assert not set(batch)&set(chosen) and len(set(batch))==len(batch)
                p.update(batch,energy[batch]);chosen+=batch;batches.append([ids[i] for i in batch])
                if first is None and min(energy[chosen])<=THRESHOLD:first=len(chosen)
                if before==4 and method in ['conservative_pi','esm_pi']:
                    pred=p.models['ridge'];mu=pred.additive+pred.posterior()[0];prob=probability(pred,p.calibration['ridge'])
                    unseen=[i for i in range(len(ids)) if i not in chosen]
                    metrics.append(dict(scheme=scheme,task_id=tid,model='old' if method=='conservative_pi' else 'esm',
                        MAE=float(np.mean(abs(mu[unseen]-energy[unseen]))),Brier=float(np.mean((prob[unseen]-(energy[unseen]<=THRESHOLD))**2))))
            runs.append(dict(scheme=scheme,fold=fold,task_id=tid,domain=t['domain'],damage=t['damage'],damage_site=site(t['damage'])+1,
                method=method,queries=len(chosen),**{f'success{k}':bool(min(energy[chosen[:k]])<=THRESHOLD) for k in [8,16,24,32]},
                first_batch_cost=first,penalized_cost=first if first else p.wt.cap+4,best_affinity=float(np.exp(-min(energy[chosen])/RT)),reachable=bool(min(energy)<=THRESHOLD)))
            stream.write(json.dumps(dict(scheme=scheme,fold=fold,task_id=tid,method=method,batches=batches))+'\n')
    return runs,metrics


def evaluate():
    check();assert (OUT/'embedding_manifest.json').exists() and not (OUT/'started.json').exists()
    dump(OUT/'started.json',dict(at_utc=datetime.now(timezone.utc).isoformat()))
    tasks=load_development();records=pd.read_csv(ROOT/'experiments/pdz2018_background_v1/historical_training_genotypes.csv.gz').to_dict('records')
    store=ESMFeatures();x=store.training(records);runs=[];metrics=[];audits=[];(OUT/'models').mkdir()
    folds=[('site',s) for s in range(9)]+[('domain',d) for d in DOMAINS]
    with gzip.open(OUT/'trajectories.jsonl.gz','wt') as stream:
        for scheme,held in folds:
            fold=f'{scheme}_{held}';print('Fitting',fold,flush=True);mask=outer_mask(records,scheme,held)
            train=[r for r,m in zip(records,mask) if m];model=Model(train,x[mask]);model.save(OUT/'models'/f'{fold}.npz')
            cal,inner=calibrate(records,x,tasks,scheme,held,mask,store)
            calibration={'old':json.loads((OLD/'models'/f'{fold}_calibration.json').read_text())['ridge'],'esm':cal}
            dump(OUT/'models'/f'{fold}_calibration.json',calibration)
            selected={tid:t for tid,t in tasks.items() if (site(t['damage'])==held if scheme=='site' else t['domain']==held)}
            trainkeys={(r['domain'],r['left'],r['right']) for r in train}
            for t in selected.values():
                for r in t['rows']:
                    a,b=sorted([t['damage'],r['id']],key=site);assert (t['domain'],a,b) not in trainkeys
            a,b=simulate(selected,store,{'old':Model.load(OLD/'models'/f'{fold}_ridge.npz'),'esm':model},calibration,scheme,fold,METHODS,stream)
            runs+=a;metrics+=b;audits.append(dict(fold=fold,training_n=len(train),features=x.shape[1],inner=inner,overlap=0))
            stream.flush();pd.DataFrame(runs).to_csv(OUT/'runs.csv',index=False);pd.DataFrame(metrics).to_csv(OUT/'prediction_metrics.csv',index=False)
            dump(OUT/'fit_audits.json',audits);print('Completed',fold,flush=True)
        siteframe=pd.DataFrame(runs).query("scheme=='site'").pivot(index='task_id',columns='method',values='success32')
        gains=int((siteframe.esm_pi & ~siteframe.conservative_pi).sum());losses=int((~siteframe.esm_pi & siteframe.conservative_pi).sum())
        do_fusion=gains>=1 and gains>=losses
        dump(OUT/'fusion_gate.json',dict(gains=gains,losses=losses,passed=do_fusion,rule='at least1 gain and no net loss vs conservative_pi, site development'))
        if do_fusion:
            for scheme,held in folds:
                fold=f'{scheme}_{held}';selected={tid:t for tid,t in tasks.items() if (site(t['damage'])==held if scheme=='site' else t['domain']==held)}
                models={'old':Model.load(OLD/'models'/f'{fold}_ridge.npz'),'esm':Model.load(OUT/'models'/f'{fold}.npz')}
                calibration=json.loads((OUT/'models'/f'{fold}_calibration.json').read_text())
                a,_=simulate(selected,store,models,calibration,scheme,fold,[FUSION],stream);runs+=a
            pd.DataFrame(runs).to_csv(OUT/'runs.csv',index=False)
    check();dump(OUT/'execution.json',dict(status='completed',trajectories=len(runs),fusion_evaluated=do_fusion,new_agnes_calls=0,sealed_domains_loaded=False))


if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='evaluate':evaluate()
    else:raise ValueError('freeze or evaluate')
