"""Independent sequence/model reconstruction and causal search audit."""
import os
os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
import gzip
import json
import numpy as np
import pandas as pd
from pdz2018_data import ROOT,RT,dump,load_development
from pdz_background_model import site,training_data
from pdz_rescue_ranking import Model,THRESHOLD,public_task,features
from pdz_esm_model import OUT,OLD,METHODS,FUSION,ESMFeatures,search,calibrate
from run_pdz_esm import check
from run_pdz_rescue_ranking import outer_mask
from prepare_pdz_esm import MODEL,variants


def embeddings():
    import torch
    from transformers import AutoTokenizer,EsmModel
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=EsmModel.from_pretrained(MODEL,local_files_only=True,add_pooling_layer=False).eval()
    projection=torch.from_numpy(np.load(OUT/'projection.npy'))
    checks=[]
    for context in json.loads((OUT/'sequence_mapping.json').read_text()):
        with np.load(OUT/f"{context['domain']}_embeddings.npz") as data:
            keys=data['keys'];pool=data['pool'];local=data['local']
        designed=list(variants(context['helix']));assert len(keys)==13168 and list(keys)==[k for k,_ in designed]
        assert np.isfinite(pool).all() and np.isfinite(local).all()
        for idx in [0,1,172,13167]:
            key=keys[idx];seq=list(context['sequence']);start=context['helix_start_0based']
            for m in ([] if key=='WT' else key.split(':')):
                pos=int(m[1:-1])-1;assert seq[start+pos]==m[0];seq[start+pos]=m[-1]
            seq=''.join(seq)
            with torch.inference_mode():
                h=model(**tokenizer(seq,return_tensors='pt')).last_hidden_state[0,1:len(seq)+1]
                actual_pool=(h.mean(0)@projection).numpy();actual_local=(h[start:start+9]@projection).numpy()
            delta=max(float(abs(actual_pool-pool[idx]).max()),float(abs(actual_local-local[idx]).max()))
            assert np.allclose(actual_pool,pool[idx],atol=.005,rtol=.001) and np.allclose(actual_local,local[idx],atol=.005,rtol=.001),(context['domain'],key,delta)
            checks.append(dict(domain=context['domain'],key=str(key),max_abs_cpu_mps_difference=delta))
    return checks


def main():
    check();assert json.loads((OUT/'execution.json').read_text())['status']=='completed'
    raw=training_data();records=pd.read_csv(ROOT/'experiments/pdz2018_background_v1/historical_training_genotypes.csv.gz').to_dict('records')
    assert [(r['domain'],r['left'],r['right']) for r in raw]==[(r['domain'],r['left'],r['right']) for r in records]
    assert np.allclose([r['epsilon'] for r in raw],[r['epsilon'] for r in records],atol=1e-12)
    embedding_checks=embeddings();print('CPU independently checked12 embeddings',flush=True)
    store=ESMFeatures();x=store.training(records);tasks=load_development();models={};calibrations={}
    for scheme,held in [('site',s) for s in range(9)]+[('domain',d) for d in ['pdz2','pdz3','syntrophin']]:
        fold=f'{scheme}_{held}';mask=outer_mask(records,scheme,held);train=[r for r,m in zip(records,mask) if m]
        models[fold]={'old':Model.load(OLD/'models'/f'{fold}_ridge.npz'),'esm':Model.load(OUT/'models'/f'{fold}.npz')}
        rebuilt=Model(train,x[mask]);assert np.allclose(rebuilt.coef,models[fold]['esm'].coef,atol=1e-10)
        assert np.isclose(rebuilt.intercept,models[fold]['esm'].intercept)
        calibrations[fold]=json.loads((OUT/'models'/f'{fold}_calibration.json').read_text())
        keys={(r['domain'],r['left'],r['right']) for r in train}
        for t in tasks.values():
            if (site(t['damage'])==held if scheme=='site' else t['domain']==held):
                for r in t['rows']:
                    a,b=sorted([t['damage'],r['id']],key=site);assert (t['domain'],a,b) not in keys
        if fold in ['site_0','domain_pdz2']:
            cal,_=calibrate(records,x,tasks,scheme,held,mask,store)
            for k,v in cal.items():assert np.isclose(v,calibrations[fold]['esm'][k],atol=1e-12)
            corrupt_records=[dict(r,epsilon=r['epsilon']+1000) if not m else r for r,m in zip(records,mask)]
            corrupt_tasks={}
            for tid,t in tasks.items():
                rows=[]
                for r in t['rows']:
                    excluded=(scheme=='site' and held in [site(t['damage']),site(r['id'])]) or (scheme=='domain' and t['domain']==held)
                    rows.append(dict(r,deltaG_kcal=r['deltaG_kcal']+1000) if excluded else r)
                corrupt_tasks[tid]={**t,'rows':rows}
            altered,_=calibrate(corrupt_records,x,corrupt_tasks,scheme,held,mask,store);assert cal==altered
        print('Verified',fold,flush=True)
    oldorders={}
    with gzip.open(OLD/'trajectories.jsonl.gz','rt') as f:
        for line in f:
            t=json.loads(line);oldorders[t['scheme'],t['task_id'],t['method']]=t['batches']
    runs=pd.read_csv(OUT/'runs.csv').set_index(['scheme','task_id','method']);assert runs.index.is_unique
    seen=set();batches=queries=old_count=0
    with gzip.open(OUT/'trajectories.jsonl.gz','rt') as f:
        for line in f:
            tr=json.loads(line);key=(tr['scheme'],tr['task_id'],tr['method']);assert key not in seen;seen.add(key)
            t=tasks[tr['task_id']];public=public_task(t);fold=tr['fold']
            base={'old':models[fold]['old'].predict(features(public)),'esm':models[fold]['esm'].predict(store.task(public))}
            p=search(public,tr['method'],base,calibrations[fold]);ids=[r['id'] for r in t['rows']]
            energy=np.array([r['deltaG_kcal'] for r in t['rows']]);chosen=[];first=None
            for expected in tr['batches']:
                batch=p.proposal();assert [ids[i] for i in batch]==expected
                assert len(batch)==min(4,p.wt.cap-len(chosen)) and not set(batch)&set(chosen)
                p.update(batch,energy[batch]);chosen+=batch;batches+=1
                if first is None and min(energy[chosen])<=THRESHOLD:first=len(chosen)
            r=runs.loc[key];assert len(chosen)==r.queries==min(32,len(ids));queries+=len(chosen)
            for n in [8,16,24,32]:assert bool(r[f'success{n}'])==bool(min(energy[chosen[:n]])<=THRESHOLD)
            assert r.penalized_cost==(first if first else p.wt.cap+4)
            assert np.isclose(r.best_affinity,np.exp(-min(energy[chosen])/RT))
            if tr['method'] in METHODS[:4]:assert tr['batches']==oldorders[key];old_count+=1
    assert seen==set(runs.index)
    gate=json.loads((OUT/'fusion_gate.json').read_text());siteframe=runs.reset_index().query("scheme=='site'").pivot(index='task_id',columns='method',values='success32')
    gains=int((siteframe.esm_pi & ~siteframe.conservative_pi).sum());losses=int((~siteframe.esm_pi & siteframe.conservative_pi).sum())
    assert gate['gains']==gains and gate['losses']==losses and gate['passed']==(gains>=1 and gains>=losses)
    assert (FUSION in runs.index.get_level_values('method'))==gate['passed']
    check()
    dump(OUT/'verification.json',dict(status='passed',genotypes_rebuilt=len(records),designed_sequences_audited=39504,
        cpu_embedding_checks=embedding_checks,outer_models_rebuilt=12,training_test_genotype_overlap=0,
        nested_calibration_corruption_invariance=['site_0','domain_pdz2'],old_trajectories_exactly_reproduced=old_count,
        causal_batches_replayed=batches,logged_queries=queries,trajectories=len(seen),fusion_gate_verified=True,
        public_only_policy_inputs=True,old_artifacts_unchanged=True,sealed_domains_loaded=False,new_agnes_calls=0))
    print('Verification passed',len(seen),'trajectories',flush=True)


if __name__=='__main__':main()
