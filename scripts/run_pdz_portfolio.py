"""Predeclared portfolios followed by nested weighted/nonlinear head comparison."""
from datetime import datetime,timezone
import gzip
import hashlib
import json
import pickle
import sys
import numpy as np
import pandas as pd
from pdz2018_data import ROOT,DOMAINS,dump,hashes,load_development
from pdz_background_model import site,pair_features
from pdz_rescue_ranking import Model,public_task,features
from pdz_esm_model import ESMFeatures
from pdz_portfolio_model import OUT,OLD,ESM,PORTFOLIOS,HEADS,METHODS,GRID,rollout,fit_head,select_configs
from run_pdz_rescue_ranking import outer_mask

FOLDS=[('site',i) for i in range(9)]+[('domain',d) for d in DOMAINS]


def freeze():
    OUT.mkdir(exist_ok=True);assert not (OUT/'freeze.json').exists()
    protocol=dict(version=OUT.name,scope='Post-result exploratory development; no independent test claim',
        tasks=207,budget=32,initial_wt=8,batch=4,endpoint='exp(-G/RT)>=0.5 WT',short_pool='exhaust min32,N; retain in denominator',
        sealed_unread=['shank3','zo1'],methods=METHODS,stages={'1':'four unchanged controls plus3 candidate portfolios',
            '2':'weighted ridge, ordinary tree and weighted tree under matched full-budget LCB; choose hyperparameters inside each outer training fold'},
        portfolios={'portfolio3':'3WT plus rotating ridge/ESM/contact, six rounds',
            'portfolio2':'2WT plus ridge plus alternating ESM/contact, six rounds',
            'portfolio1':'WT/ridge/ESM/contact each nominate one per batch'},
        nomination='Each expert ordered list from shared actually queried feedback; first unqueried unchosen candidate; contact positive distance/chemical order, uniform seeded fallback; no old-trajectory stitching or unqueried label input',
        training_access='Same32851 historical canonical development genotypes as previous models; ESM portfolios reuse frozen full-construct representations',
        folds='Nine site folds exclude both endpoints across all3 domains;3 leave-domain-out folds',
        weighting='Based ONLY on public singles: weightw if either single affinity<=.2, otherwise1; normalize mean1; w chosen inside training folds',
        model_grid=GRID,linear='same204 structural features, unweighted feature standardization, ridgealpha10; weighted squared loss; free intercept',
        tree='same features and residual target; HistGradientBoostingRegressor,150 rounds,lr.05,minleaf40,L2=10,no early stopping,seed20261008',
        inner='3 held site groups modulo3; exclude inner sites from both training endpoints, additionally enforce outer exclusions in training AND validation task candidate pools',
        inner_selection='Max total Success@32 over all eligible inner tasks, then min summed penalized first-batch cost, then shallower depth/lower weight; exactly8 configurations, no outer-score tuning',
        search='New heads use unchanged structural RBF GP,noise.09,LCB mean-minus-one-SD and4 full-budget suggestions after shared8; entire batch before feedback',
        outer_selection='Max site success32 among6 new methods, tie lower penalized first-batch cost, then method order. Transfer is a subsequent gate, not retuning',
        comparisons=[['portfolio3','contact_seed0'],['portfolio2','portfolio3'],['portfolio1','portfolio2'],
            ['weighted_ridge','ridge_full_lcb'],['tree','ridge_full_lcb'],['weighted_tree','tree']],
        interaction_limit='Tree depths selected separately within folds; comparisons concern nested model procedures, not isolated fixed-depth weight effects',
        agnes_gate=dict(net_site_vs_best_original_at_least=3,net_transfer_vs_best_original_at_least=1,gained_sites_vs_WT_at_least=2,worst_transfer_domain_vs_WT_at_least=-1),
        agnes_if_pass='Fresh32-task original identity cohort, same chosen predictor and full legal candidate information, same32 query budget. No sealed labels',
        metrics=['Success@8/16/24/32','paired gains/losses','domain results','penalized first completed batch cost; failure cap+4','shared8 residual MAE','descriptive grouped bootstrap'],
        limitations=['Previously inspected development data','Only3 domains and9 sites','point-only; missing biological uncertainty',
          'ESM uses reference PDB constructs, not verified full assay plasmids','Grouped intervals descriptive because genotypes couple multiple damage tasks'])
    dump(OUT/'protocol.json',protocol)
    preserved={}
    for folder in ['pdz2018_background_v1','pdz2018_rescue_ranking_v1','pdz2018_esm_v2','pdz2018_point_dev_v1','pdz2018_uncertainty_v1','gb1_final_holdout_v1']:
        preserved.update(hashes([p for p in (ROOT/'experiments'/folder).rglob('*') if p.is_file()]))
    dump(OUT/'preserved_hashes.json',preserved)
    names=['pdz_portfolio_model.py','run_pdz_portfolio.py','verify_pdz_portfolio.py','report_pdz_portfolio.py',
           'pdz_esm_model.py','pdz_rescue_ranking.py','pdz_background_model.py','pdz2018_data.py','pdz2018_policy.py','prepare_pdz_esm.py','run_pdz_rescue_ranking.py']
    paths=[ROOT/'scripts'/n for n in names]+[OUT/'protocol.json',OUT/'preserved_hashes.json',
        ROOT/'data/pdz2018/source/Salinas_Ranganathan_data.mat',ROOT/'data/public_reference/residue_volumes.json',
        ROOT/'experiments/dataset_feasibility_v1/pdz2018_calibration_QC.csv']
    paths+=list((ROOT/'data/pdz2018/public').glob('*_geometry.npz'))
    dump(OUT/'freeze.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes(paths)))


def check():
    for name in ['freeze.json','preserved_hashes.json']:
        obj=json.loads((OUT/name).read_text())
        for p,h in obj.get('hashes',obj).items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p


def select_tasks(tasks,scheme,held):
    return {tid:t for tid,t in tasks.items() if (site(t['damage'])==held if scheme=='site' else t['domain']==held)}


def execute_tasks(tasks,bases,methods,scheme,fold,stream):
    runs=[]
    for tid,t in tasks.items():
        for method in methods:
            stats,batches=rollout(t,method,bases[tid])
            runs.append(dict(scheme=scheme,fold=fold,task_id=tid,domain=t['domain'],damage=t['damage'],damage_site=site(t['damage'])+1,method=method,**stats))
            stream.write(json.dumps(dict(scheme=scheme,fold=fold,task_id=tid,method=method,batches=batches))+'\n')
    return runs


def evaluate():
    check();assert not (OUT/'started.json').exists()
    dump(OUT/'started.json',dict(at_utc=datetime.now(timezone.utc).isoformat()))
    tasks=load_development();assert len(tasks)==207
    store=ESMFeatures();runs=[];metrics=[]
    (OUT/'models').mkdir();(OUT/'selection').mkdir()
    with gzip.open(OUT/'trajectories.jsonl.gz','wt') as stream:
        # First complete the cheap, no-new-training portfolio stage.
        for scheme,held in FOLDS:
            fold=f'{scheme}_{held}';selected=select_tasks(tasks,scheme,held)
            old=Model.load(OLD/'models'/f'{fold}_ridge.npz');esm=Model.load(ESM/'models'/f'{fold}.npz')
            bases={tid:{'ridge':old.predict(features(public_task(t))),'esm':esm.predict(store.task(public_task(t)))} for tid,t in selected.items()}
            runs+=execute_tasks(selected,bases,METHODS[:7],scheme,fold,stream)
            stream.flush();pd.DataFrame(runs).to_csv(OUT/'runs.csv',index=False)
            print('Portfolio completed',fold,flush=True)
        dump(OUT/'portfolio_completed.json',dict(trajectories=len(runs),at_utc=datetime.now(timezone.utc).isoformat()))
        del store
        records=pd.read_csv(ROOT/'experiments/pdz2018_background_v1/historical_training_genotypes.csv.gz').to_dict('records')
        x=np.array([pair_features(r['left'],r['right'],r['gd'],r['gr'],r['distance'],True) for r in records])
        for scheme,held in FOLDS:
            fold=f'{scheme}_{held}';print('Nested selection',fold,flush=True);mask=outer_mask(records,scheme,held)
            configs,audit=select_configs(records,x,tasks,scheme,held,mask)
            dump(OUT/'selection'/f'{fold}.json',dict(configs=configs,audit=audit))
            train=[r for r,m in zip(records,mask) if m];models={k:fit_head(train,x[mask],configs[k]) for k in HEADS}
            with (OUT/'models'/f'{fold}.pkl').open('wb') as f:pickle.dump(models,f,protocol=5)
            selected=select_tasks(tasks,scheme,held);trainkeys={(r['domain'],r['left'],r['right']) for r in train}
            bases={}
            for tid,t in selected.items():
                for r in t['rows']:
                    a,b=sorted([t['damage'],r['id']],key=site);assert (t['domain'],a,b) not in trainkeys
                xf=features(public_task(t));bases[tid]={k:m.predict(xf) for k,m in models.items()}
                from pdz_background_model import OnlinePredictor
                order=np.lexsort((np.array([r['id'] for r in t['rows']]),-np.array([r['WT_log_affinity'] for r in t['rows']])))
                energy=np.array([r['deltaG_kcal'] for r in t['rows']]);initial=list(map(int,order[:8]));unseen=order[8:]
                for k,base in bases[tid].items():
                    p=OnlinePredictor(public_task(t),base,True);p.update(initial,energy[initial]-p.additive[initial]);mu=p.additive+p.posterior()[0]
                    metrics.append(dict(scheme=scheme,task_id=tid,method=k,MAE=float(np.mean(abs(mu[unseen]-energy[unseen])))))
            runs+=execute_tasks(selected,bases,HEADS,scheme,fold,stream);stream.flush()
            pd.DataFrame(runs).to_csv(OUT/'runs.csv',index=False);pd.DataFrame(metrics).to_csv(OUT/'prediction_metrics.csv',index=False)
            print('Heads completed',fold,json.dumps(configs),flush=True)
    check();dump(OUT/'execution.json',dict(status='completed',trajectories=len(runs),new_agnes_calls=0,sealed_domains_loaded=False))


if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='evaluate':evaluate()
    else:raise ValueError('freeze or evaluate')
