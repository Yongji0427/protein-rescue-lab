"""Verify all budgets/metrics plus causal replay from identity-only catalogs."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from experiment_data import ROOT
from factor_policy import FactorPolicy


def main():
    out = ROOT/sys.argv[1]; p=json.loads((out/'protocol.json').read_text())
    for path,digest in p['hashes'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    log=pd.read_csv(out/'query_log.csv.gz'); runs=pd.read_csv(out/'runs.csv')
    keys=['damage','method','repeat']; sizes=log.groupby(keys).size()
    assert (sizes==32).all() and not log.duplicated(keys+['rescue']).any()
    assert set(log.damage)==set(p['damage_tasks'])
    dp=log.damage.str[1:-1].astype(int); rp=log.rescue.str[1:-1].astype(int)
    held=p['holdout_full_sequence_positions']; assert not rp.isin(held).any()
    assert dp.isin(held).all() if p['split']=='holdout' else not dp.isin(held).any()
    assert (dp!=rp).all() and (log.batch_cost==4*((log['query']+3)//4)).all()
    catalog=pd.read_csv(ROOT/'data/evaluator_only/gb1_raw.csv',usecols=['mutant','Mut1 Position','Mut2 Position'])
    a,b=catalog['Mut1 Position']+226,catalog['Mut2 Position']+226
    mask=b.notna() & ((a.isin(held)^b.isin(held)) if p['split']=='holdout' else (~a.isin(held)&~b.isin(held)))
    catalog=catalog.loc[mask].copy();parts=catalog.mutant.str.split(':',expand=True);catalog['a'],catalog['b']=parts[0],parts[1]
    g1={k:g for k,g in catalog.groupby('a',sort=False)};g2={k:g for k,g in catalog.groupby('b',sort=False)};empty=catalog.iloc[:0]
    singles=pd.read_csv(ROOT/'data/public_reference/singles.csv').set_index('mutant');batches=0
    for (damage,method), rows in log[log.repeat==0].groupby(['damage','method']):
        pool=pd.concat([g1.get(damage,empty),g2.get(damage,empty)])
        ids=np.sort(np.where(pool.a==damage,pool.b,pool.a))
        seed=np.random.SeedSequence([p['seed'],int(damage[1:-1]),ord(damage[-1]),0])
        policy=FactorPolicy(ids,singles.loc[ids,'lnW'].to_numpy(),float(singles.loc[damage,'lnW']),damage,method,seed)
        for cost, batch in rows.groupby('batch_cost',sort=True):
            batch=batch.sort_values('query'); prediction=policy.predict();choices=policy.choose_batch();indices=[i for i,_ in choices]
            assert list(ids[indices])==batch.rescue.tolist()
            assert [slot for _,slot in choices]==batch.slot.tolist()
            assert np.allclose(prediction[indices],batch.predicted_lnW_before_batch)
            if cost<=8: assert indices==list(policy.order[cost-4:cost])
            if method.endswith('gate_random') and cost>8 and batch.slot.iloc[-1]=='gate_random':
                avail=np.ones(len(ids),dtype=bool);avail[policy.observed+indices[:3]]=False
                if np.any(avail&(policy.contact_score>0)): assert policy.contact_score[indices[-1]]>0
            policy.update(indices,batch.lnW.tolist());batches+=1
    for budget in range(4,33,4):
        part=log[log['query']<=budget].groupby(keys)
        observed=part.W_display.max(); lower=part.count_ci_low.max()
        compared=runs[runs.budget==budget].set_index(keys)
        initial = singles.loc[compared.index.get_level_values('damage'),'W_display'].to_numpy()
        assert np.allclose(compared.best_W,np.maximum(initial,observed.reindex(compared.index).to_numpy()))
        assert np.array_equal(compared.success.to_numpy(),observed.reindex(compared.index).to_numpy()>=.5)
        assert np.array_equal(compared.count_robust_success.to_numpy(),lower.reindex(compared.index).to_numpy()>=.5)
    # No-feedback scores cannot depend on revealed labels; gate-empty factor shares fallback draws.
    ids=['K254A','K257Y','V280T','V280W','A249F','A250Y','L238A','F278A','Y259A','Y259F','V265G','V265W']
    policies=[FactorPolicy(ids,np.arange(12.),-3.,'K254W',method,7) for method in p['methods'] if method!='wt_rank']
    for obj in policies: obj.update([0,1,2,3],[9.,-8.,2.,-5.]);obj.update([4,5,6,7],[1.,2.,3.,4.])
    if policies:
        assert np.array_equal(policies[0].predict(),policies[0].prior)
        if len(policies)>1: assert policies[0].choose_batch()==policies[1].choose_batch()
    result=dict(status='passed',queries=len(log),causal_replay_batches=batches,split=p['split'],checks=['all frozen input and code hashes match','all runs use 32 unique legal queries; identical WT first8; no feedback inside a committed batch','seed-zero causal replay of every method/task using only identity catalog, public singles and prior queried labels','all budget metrics independently reconstructed from query log','no-feedback prediction independent of outcomes; chemical gate matched; gate-empty fallback identical'])
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
