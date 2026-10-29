"""Audit nested selection boundaries and replay every shared-feedback query."""
import gzip
import json
import pickle
import numpy as np
import pandas as pd
from pdz2018_data import ROOT,dump,load_development
from pdz_background_model import site,training_data,pair_features
from pdz_rescue_ranking import Model,public_task,features
from pdz_esm_model import ESMFeatures
from pdz_portfolio_model import OUT,OLD,ESM,PORTFOLIOS,HEADS,METHODS,GRID,fit_head,select_configs,rollout
from run_pdz_portfolio import check,FOLDS,select_tasks
from run_pdz_rescue_ranking import outer_mask


def main():
    check();assert json.loads((OUT/'execution.json').read_text())['status']=='completed'
    tasks=load_development();records=pd.read_csv(ROOT/'experiments/pdz2018_background_v1/historical_training_genotypes.csv.gz').to_dict('records');raw=training_data()
    assert [(r['domain'],r['left'],r['right']) for r in records]==[(r['domain'],r['left'],r['right']) for r in raw]
    assert np.allclose([r['epsilon'] for r in raw],[r['epsilon'] for r in records],atol=1e-12)
    x=np.array([pair_features(r['left'],r['right'],r['gd'],r['gr'],r['distance'],True) for r in records]);assert x.shape[1]==204
    store=ESMFeatures();models={};old_models={};esm_models={}
    for scheme,held in FOLDS:
        fold=f'{scheme}_{held}';mask=outer_mask(records,scheme,held);train=[r for r,m in zip(records,mask) if m]
        selection=json.loads((OUT/'selection'/f'{fold}.json').read_text());configs=selection['configs']
        # Check every selected config follows the frozen ranking of the complete grid.
        for family,entries in selection['audit']['grid_scores'].items():
            assert [e['config'] for e in entries]==GRID[family]
            assert len({e['tasks'] for e in entries})==1
            best=sorted(entries,key=lambda e:(-e['successes'],e['cost'],e['config']['depth'],e['config']['weight']))[0]
            assert best['config']==configs[family]
        with (OUT/'models'/f'{fold}.pkl').open('rb') as f:models[fold]=pickle.load(f)
        for k in HEADS:
            rebuilt=fit_head(train,x[mask],configs[k])
            for tid,t in select_tasks(tasks,scheme,held).items():
                xf=features(public_task(t));assert np.allclose(rebuilt.predict(xf),models[fold][k].predict(xf),atol=1e-10)
        if fold in ['site_0','domain_pdz2']:
            selected,audit=select_configs(records,x,tasks,scheme,held,mask)
            assert selected==configs and audit==selection['audit']
            corrupt=[dict(r,epsilon=r['epsilon']+1000) if not m else r for r,m in zip(records,mask)]
            altered_tasks={}
            for tid,t in tasks.items():
                rows=[]
                for r in t['rows']:
                    excluded=(scheme=='site' and held in [site(t['damage']),site(r['id'])]) or (scheme=='domain' and t['domain']==held)
                    rows.append(dict(r,deltaG_kcal=r['deltaG_kcal']+1000) if excluded else r)
                altered_tasks[tid]={**t,'rows':rows}
            s2,a2=select_configs(corrupt,x,altered_tasks,scheme,held,mask)
            assert selected==s2 and audit==a2,fold
        keys={(r['domain'],r['left'],r['right']) for r in train}
        for t in select_tasks(tasks,scheme,held).values():
            for r in t['rows']:
                a,b=sorted([t['damage'],r['id']],key=site);assert (t['domain'],a,b) not in keys
        old_models[fold]=Model.load(OLD/'models'/f'{fold}_ridge.npz');esm_models[fold]=Model.load(ESM/'models'/f'{fold}.npz')
        print('Verified nested fold',fold,flush=True)
    old_traces={}
    with gzip.open(ESM/'trajectories.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);old_traces[r['scheme'],r['task_id'],r['method']]=r['batches']
    runs=pd.read_csv(OUT/'runs.csv').set_index(['scheme','task_id','method']);assert len(runs)==2*207*len(METHODS) and runs.index.is_unique
    seen=set();batches=queries=old_count=0;cache={}
    with gzip.open(OUT/'trajectories.jsonl.gz','rt') as f:
        for line in f:
            tr=json.loads(line);key=(tr['scheme'],tr['task_id'],tr['method']);assert key not in seen;seen.add(key)
            fold,tid=tr['fold'],tr['task_id'];t=tasks[tid]
            if (fold,tid) not in cache:
                public=public_task(t);xf=features(public)
                cache[fold,tid]={'ridge':old_models[fold].predict(xf),'esm':esm_models[fold].predict(store.task(public)),
                    **{k:m.predict(xf) for k,m in models[fold].items()}}
            stats,actual=rollout(t,tr['method'],cache[fold,tid]);assert actual==tr['batches'];row=runs.loc[key]
            for n in [8,16,24,32]:assert stats[f'success{n}']==bool(row[f'success{n}'])
            assert stats['queries']==row.queries==min(32,len(t['rows'])) and stats['penalized_cost']==row.penalized_cost
            assert np.isclose(stats['best_affinity'],row.best_affinity)
            if tr['method'] in METHODS[:4]:assert actual==old_traces[key];old_count+=1
            flat=[r for batch in actual for r in batch];assert len(flat)==len(set(flat))
            batches+=len(actual);queries+=len(flat)
    assert seen==set(runs.index);check()
    dump(OUT/'verification.json',dict(status='passed',source_genotypes_rebuilt=len(records),outer_heads_rebuilt=36,
        inner_grid_configurations=8,nested_selection_rebuilt_and_outer_label_corruption_invariant=['site_0','domain_pdz2'],
        training_outer_genotype_overlap=0,causal_batches_replayed=batches,logged_queries=queries,trajectories=len(seen),
        old_trajectories_exactly_reproduced=old_count,all_task_budgets_verified=True,old_artifacts_unchanged=True,
        sealed_domains_loaded=False,new_agnes_calls=0))
    print('VERIFIED',len(seen),'trajectories',flush=True)


if __name__=='__main__':main()
