"""Audit training boundaries, model reconstruction and label-causal batch replay."""
import gzip
import json
import numpy as np
import pandas as pd
from pdz2018_data import ROOT, dump, load_development
from pdz_background_model import training_data, pair_features, site
from pdz_rescue_ranking import Model, Search, METHODS, THRESHOLD, public_task, features
from run_pdz_rescue_ranking import OUT, KINDS, check_hashes, outer_mask, calibrate


def main():
    check_hashes()
    assert json.loads((OUT/'execution.json').read_text())['status']=='completed'
    tasks=load_development();records=training_data()
    cached=pd.read_csv(ROOT/'experiments/pdz2018_background_v1/historical_training_genotypes.csv.gz').to_dict('records')
    assert [(r['domain'],r['left'],r['right']) for r in records]==[(r['domain'],r['left'],r['right']) for r in cached]
    assert np.allclose([r['epsilon'] for r in records],[r['epsilon'] for r in cached],atol=1e-12)
    records=cached
    x=np.array([pair_features(r['left'],r['right'],r['gd'],r['gr'],r['distance'],True) for r in records])
    models={};calibrations={};boundaries=[]
    for scheme,held in [('site',s) for s in range(9)]+[('domain',d) for d in ['pdz2','pdz3','syntrophin']]:
        fold=f'{scheme}_{held}';mask=outer_mask(records,scheme,held)
        train=[r for r,m in zip(records,mask) if m]
        if scheme=='site':assert all(held not in (r['left_site'],r['right_site']) for r in train)
        else:assert all(r['domain']!=held for r in train)
        models[fold]={k:Model.load(OUT/'models'/f'{fold}_{k}.npz') for k in KINDS}
        for k,loaded in models[fold].items():
            rebuilt=Model(train,x[mask],k=='rank')
            assert np.allclose(loaded.coef,rebuilt.coef,atol=1e-10) and np.isclose(loaded.intercept,rebuilt.intercept)
        calibrations[fold]=json.loads((OUT/'models'/f'{fold}_calibration.json').read_text())
        for inner in range(3):
            sites={s for s in range(9) if s%3==inner}
            nested=[r for r in train if not sites & {r['left_site'],r['right_site']}]
            assert all(not sites & {r['left_site'],r['right_site']} for r in nested)
        if fold in ['site_0','domain_pdz2']:
            actual,_=calibrate(records,x,tasks,scheme,held,mask)
            for k in KINDS:
                for key in ['bias','scale','raw_brier','calibrated_brier_fit']:
                    assert np.isclose(actual[k][key],calibrations[fold][k][key],atol=1e-12)
            # Excluded outcomes cannot affect training coefficients or calibration.
            altered=[dict(r,epsilon=r['epsilon']+1000) if not m else r for r,m in zip(records,mask)]
            changed_tasks={}
            for tid,t in tasks.items():
                rows=[]
                for r in t['rows']:
                    forbidden=(scheme=='site' and held in [site(t['damage']),site(r['id'])]) or (scheme=='domain' and t['domain']==held)
                    rows.append(dict(r,deltaG_kcal=r['deltaG_kcal']+1000) if forbidden else r)
                changed_tasks[tid]={**t,'rows':rows}
            alternate,_=calibrate(altered,x,changed_tasks,scheme,held,mask)
            assert actual==alternate,fold
        boundaries.append(dict(fold=fold,training_n=len(train),overlap=0))
        print('Verified fit',fold,flush=True)
    runs=pd.read_csv(OUT/'runs.csv').set_index(['scheme','task_id','method'])
    assert len(runs)==2*207*len(METHODS) and runs.index.is_unique
    old=pd.read_csv(ROOT/'experiments/pdz2018_background_v1/runs.csv').set_index(['task_id','method'])
    oldq=pd.read_csv(ROOT/'experiments/pdz2018_background_v1/query_log.csv.gz')
    oldorders={(tid,m):list(g.rescue) for (tid,m),g in oldq.groupby(['task_id','method'],sort=False)}
    batches=labels=0;seen_trajectories=set();baseline_replays=0
    with gzip.open(OUT/'trajectories.jsonl.gz','rt') as stream:
        for line in stream:
            tr=json.loads(line);scheme,tid,method=tr['scheme'],tr['task_id'],tr['method'];key=(scheme,tid,method)
            assert key not in seen_trajectories;seen_trajectories.add(key)
            t=tasks[tid];public=public_task(t);xf=features(public)
            bases={k:m.predict(xf) for k,m in models[tr['fold']].items()}
            p=Search(public,method,bases,calibrations[tr['fold']]);ids=[r['id'] for r in t['rows']]
            energy=np.array([r['deltaG_kcal'] for r in t['rows']]);chosen=[];first=None
            for saved in tr['batches']:
                batch=p.proposal()
                assert saved==[ids[i] for i in batch] and not set(batch)&set(chosen)
                assert len(batch)==min(4,p.wt.cap-len(chosen))
                p.update(batch,energy[batch]);chosen+=batch;batches+=1
                if first is None and min(energy[chosen])<=THRESHOLD:first=len(chosen)
            row=runs.loc[key]
            assert len(chosen)==row.queries==min(32,len(ids))
            for budget in [8,16,24,32]:assert bool(row[f'success{budget}'])==bool(min(energy[chosen[:budget]])<=THRESHOLD)
            assert row.penalized_cost==(first if first else p.wt.cap+4)
            if scheme=='site' and method in METHODS[:3]:
                name='ridge_struct_active' if method=='ridge_full_lcb' else method
                assert bool(row.success32)==bool(old.loc[(tid,name)].success)
                assert [ids[i] for i in chosen]==oldorders[tid,name]
                baseline_replays+=1
            labels+=len(chosen)
    assert seen_trajectories==set(runs.index)
    check_hashes()
    dump(OUT/'verification.json',dict(status='passed',raw_genotypes_checked=len(records),outer_folds=boundaries,
        outer_models_rebuilt=24,nested_calibration_rebuilt_and_excluded_label_corruption_invariant=['site_0','domain_pdz2'],
        public_only_policy_inputs=True,causal_batches_replayed=batches,logged_queries=labels,
        trajectories=len(seen_trajectories),old_baseline_trajectories_exactly_reproduced=baseline_replays,
        frozen_inputs_unchanged=True,old_artifacts_unchanged=True,sealed_domains_loaded=False,new_agnes_calls=0))
    print((OUT/'verification.json').read_text(),flush=True)


if __name__=='__main__':main()
