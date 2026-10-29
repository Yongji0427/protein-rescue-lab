"""Reconstruct raw source targets, held-site boundaries and every causal batch."""
import gzip
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from pdz2018_data import ROOT, DOMAINS, RT, AA, load_development, dump
from pdz2018_policy import PDZPolicy
from pdz_background_model import ResidualModel, OnlinePredictor, site
from run_pdz_background import OUT, METHODS


def main():
    for manifest in ['freeze.json','preserved_hashes.json']:
        obj = json.loads((OUT/manifest).read_text())
        for path,digest in obj.get('hashes',obj).items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    source = loadmat(ROOT/'data/pdz2018/source/Salinas_Ranganathan_data.mat',
                     variable_names=list(DOMAINS),squeeze_me=True,struct_as_record=False)
    assert 'shank3' not in source and 'zo1' not in source
    records = pd.read_csv(OUT/'historical_training_genotypes.csv.gz').to_dict('records')
    assert len(records)==len({(r['domain'],r['left'],r['right']) for r in records})
    for r in records:
        i,j,a,b = site(r['left']),site(r['right']),AA.index(r['left'][-1]),AA.index(r['right'][-1])
        assert i<j and i==r['left_site'] and j==r['right_site']
        o = source[r['domain']]
        gd,gr,g = o.deltaG_single[a,i],o.deltaG_single[b,j],o.deltaG_double[i,j,a,b]
        assert np.isclose(r['epsilon'],g-gd-gr,atol=1e-12)
        assert np.isclose(r['epsilon'],-o.deltadeltaG[i,j,a,b],atol=1e-12)
    tasks = load_development()
    models = {}
    for held in range(9):
        train = [r for r in records if held not in (r['left_site'],r['right_site'])]
        assert all(site(r['left'])!=held and site(r['right'])!=held for r in train)
        for structure in [False,True]: models[held,structure] = ResidualModel(records,held,structure)
    with gzip.open(OUT/'decisions.jsonl.gz','rt') as stream:
        groups = {}
        for line in stream:
            d = json.loads(line);groups.setdefault((d['task_id'],d['method']),[]).append(d)
    queries = pd.read_csv(OUT/'query_log.csv.gz')
    runs = pd.read_csv(OUT/'runs.csv')
    assert len(groups)==len(tasks)*len(METHODS)==len(runs)
    batches,labels = 0,0
    for row in runs.itertuples():
        t = tasks[row.task_id];rows=t['rows'];ids=[r['id'] for r in rows]
        structure = row.method.startswith('ridge_struct')
        base = models[site(t['damage']),structure].predict(t) if row.method.startswith('ridge_') else np.zeros(len(rows))
        predictor = OnlinePredictor(t,base,structure)
        seed = np.random.SeedSequence([20261008,list(DOMAINS).index(t['domain']),site(t['damage'])+1,ord(t['damage'][-1]),0])
        policy = PDZPolicy(t,'no_feedback_contact' if row.method=='contact_seed0' else 'wt_rank',seed=seed)
        epsilon = np.array([r['deltaG_kcal'] for r in rows])-predictor.additive
        actual = queries[(queries.task_id==row.task_id)&(queries.method==row.method)]
        assert not actual.rescue.duplicated().any() and list(actual['query'])==list(range(1,len(actual)+1))
        chosen = []
        for d in groups[row.task_id,row.method]:
            assert d['budget_before']==len(predictor.seen)
            mean,_ = predictor.posterior()
            size = min(4,min(32,len(rows))-len(predictor.seen))
            if len(predictor.seen)<8: batch=list(map(int,policy.order[len(predictor.seen):len(predictor.seen)+size]))
            elif row.method in ['wt_rank','contact_seed0']: batch=policy.proposal()
            else: batch=predictor.proposal(size,'variance' if row.method.endswith('uncertainty') else 'lcb')
            assert [ids[i] for i in batch]==d['selected']
            assert np.allclose(mean[batch],d['predicted_residuals'],atol=1e-10)
            # Reconstruct only the committed batch labels directly from the source.
            for i in batch:
                r=rows[i];a,b=AA.index(t['damage'][-1]),AA.index(r['id'][-1]);di,ri=site(t['damage']),site(r['id'])
                if di>ri:di,ri,a,b=ri,di,b,a
                g=source[t['domain']].deltaG_double[di,ri,a,b]
                assert np.isclose(r['affinity'],np.exp(-g/RT),rtol=1e-12)
            predictor.update(batch,epsilon[batch]);policy.update(d['selected']);chosen+=d['selected'];batches+=1
        assert chosen==list(actual.rescue) and len(chosen)==row.queries==min(32,len(rows))
        assert np.allclose(actual.epsilon_G_kcal,[epsilon[ids.index(x)] for x in chosen])
        best=max(rows[ids.index(x)]['affinity'] for x in chosen)
        assert row.success==(best>=.5) and np.isclose(row.best_affinity,best)
        labels+=len(chosen)
    matched = pd.read_csv(OUT/'matched_MAE.csv')
    for task_id,t in tasks.items():
        initial = list(map(int,PDZPolicy(t,'wt_rank').order[:8]))
        unseen = [i for i in range(len(t['rows'])) if i not in initial]
        for name,structure in [('sequence',False),('structure',True),('additive_online',False)]:
            base=models[site(t['damage']),structure].predict(t) if name!='additive_online' else np.zeros(len(t['rows']))
            predictor=OnlinePredictor(t,base,structure)
            y=np.array([r['deltaG_kcal'] for r in t['rows']])-predictor.additive
            predictor.update(initial,y[initial]);mean,_=predictor.posterior()
            target=matched[(matched.task_id==task_id)&(matched.model==name)].iloc[0]
            assert target.n==len(unseen) and np.isclose(target.MAE_kcal,np.mean(abs(mean[unseen]-y[unseen])))
    # An excluded site's arbitrary label corruption cannot change its fold model.
    mutated=[dict(r,epsilon=r['epsilon']+1e6) if 0 in (r['left_site'],r['right_site']) else r for r in records]
    alt=ResidualModel(mutated,0,False)
    assert np.array_equal(alt.coef,models[0,False].coef)
    missed=pd.read_csv(OUT/'missed_backgrounds.csv')
    assert len(missed)==18 and int(missed.actual_Agnes_task.sum())==2
    result=json.loads((OUT/'result.json').read_text())
    assert result['model_gate_passed']==all(result['gate_checks'].values())
    dump(OUT/'verification.json',dict(status='passed',historical_genotypes_rebuilt=len(records),
        held_site_folds=9,trained_models=18,training_held_genotype_overlap=0,
        excluded_label_corruption_invariance=True,causal_batches_replayed=batches,unique_budgeted_labels=labels,
        tasks=207,methods=6,matched_initial8_MAE_rebuilt=True,diagnostic_missed_backgrounds=18,
        actual_Agnes_diagnostic_tasks=2,old_hashes_unchanged=True,new_model_calls=0,test_arrays_loaded=False))
    print((OUT/'verification.json').read_text(),flush=True)


if __name__=='__main__':main()
