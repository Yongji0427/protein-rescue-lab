"""Audit actual AGH traces, replay public menus and compare identical task sets."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from agnes_client import read_config
from experiment_data import ROOT
from factor_policy import FactorPolicy
from menu_policy import make_menu
from run_pilot import canonical_pair, count_interval


def main():
    out=ROOT/sys.argv[1];p=json.loads((out/'protocol.json').read_text())
    singles=pd.read_csv(ROOT/'data/public_reference/singles.csv').set_index('mutant')
    raw=pd.read_csv(ROOT/'data/evaluator_only/gb1_raw.csv')
    raw=raw[raw['Mut2 Position'].notna()].copy()
    raw['pair']=raw.mutant.map(lambda x: ':'.join(sorted(x.split(':'),key=lambda r:int(r[1:-1]))))
    # Reachability is not computed here. Only queried labels and score-free identities are consumed.
    catalog=raw[['mutant','Mut1 Position','Mut2 Position']].copy()
    held=p['holdout_full_sequence_positions'];a=catalog['Mut1 Position']+226;b=catalog['Mut2 Position']+226
    mask=(a.isin(held)^b.isin(held)) if p['split']=='holdout' else (~a.isin(held)&~b.isin(held))
    catalog=catalog[mask].copy();parts=catalog.mutant.str.split(':',expand=True);catalog['a'],catalog['b']=parts[0],parts[1]
    g1={k:g for k,g in catalog.groupby('a',sort=False)};g2={k:g for k,g in catalog.groupby('b',sort=False)};empty=catalog.iloc[:0]
    raw=raw.set_index('pair');key=read_config()['AGNES_API_KEY']
    records=[];queries=[];ledger=[];warnings=[];decisions=[];sessions=[];replay=0;tool_calls=0;metadata_calls=[]
    for damage in p['damage_tasks']:
        pool=pd.concat([g1.get(damage,empty),g2.get(damage,empty)])
        ids=np.sort(np.where(pool.a==damage,pool.b,pool.a));wt=singles.loc[ids,'lnW'].to_numpy()
        trial=json.loads((out/'agh_events'/(damage+'.run.json')).read_text());sessions.append(trial['session_id'])
        text=(out/'agh_events'/(damage+'.jsonl')).read_text();assert key not in text
        events=list({json.loads(line)['id']:json.loads(line) for line in text.splitlines()}.values())
        calls=[e for e in events if e['type']=='tool/call'];tool_calls+=len(calls)
        for call in calls:
            if call['data']['name']=='tool_describe' and call['data']['args']=={'name':'query_rescue_batch'}:
                metadata_calls.append({'damage':damage,'tool':'tool_describe','target':'query_rescue_batch','scope':'schema only, no assay data'})
            else: assert call['data']['name']=='query_rescue_batch','Unrelated data-access tool violates protocol'
        for event in events:
            if event['type']=='request/sent':assert event['data']['model']['id']==p['model']
            if event['type']=='cost/ledger':
                assert event['data']['model']==p['model'];ledger.append({'damage':damage,**event['data']})
        for method,folder in [('Agnes_structure','queried_results'),('fixed_same_menu','fixed_menu_results')]:
            state=json.loads((out/folder/(damage+'.json')).read_text());rows=state['queries']
            assert 8<=len(rows)<=32 and len({r['rescue'] for r in rows})==len(rows)
            seed=np.random.SeedSequence([p['seed'],int(damage[1:-1]),ord(damage[-1]),0])
            policy=FactorPolicy(ids,wt,float(singles.loc[damage,'lnW']),damage,p['selected_method'],seed)
            index={str(r):i for i,r in enumerate(ids)}
            for start in [0,4]:
                batch=policy.choose_batch();indices=[i for i,_ in batch]
                assert list(ids[indices])==[r['rescue'] for r in rows[start:start+4]]
                policy.update(indices,[r['lnW'] for r in rows[start:start+4]])
            for decision in state['decisions']:
                menu=make_menu(policy);assert menu==decision['menu']
                selected=decision['selected'];allowed={c['id'] for c in menu['candidates']}
                assert len(selected)==len(set(selected))==4 and set(selected)<=allowed
                assert menu['required_WT_anchor'] in selected
                assert set(selected).isdisjoint(set(ids[policy.observed]))
                if method=='fixed_same_menu':assert selected==menu['fixed_rule_proposal']
                else:
                    assert any(e['data']['args'].get('rescue_ids')==selected and e['data']['args'].get('task_token')==json.loads((out/'public_tasks'/(damage+'.json')).read_text())['task_token'] for e in calls)
                    decisions.append({'damage':damage,**decision})
                indices=[index[r] for r in selected];before=decision['budget_before']
                assert before==len(policy.observed)
                feedback=rows[before:before+4];assert [r['rescue'] for r in feedback]==selected
                policy.update(indices,[r['lnW'] for r in feedback]);replay+=1
            assert len(policy.observed)==len(rows)
            pairs=[canonical_pair(damage,r['rescue']) for r in rows]
            measured=raw.loc[pairs];lo,_=count_interval(measured)
            assert np.allclose([r['lnW'] for r in rows],measured.lnW)
            assert np.allclose([r['count_ci_low'] for r in rows],lo)
            assert all(int(r['rescue'][1:-1]) not in held for r in rows)
            assert int(damage[1:-1]) in held if p['split']=='holdout' else int(damage[1:-1]) not in held
            if method=='Agnes_structure' and len(rows)<32:warnings.append({'damage':damage,'queries':len(rows)})
            for row in rows:queries.append({'damage':damage,'method':method,**row})
            for budget in [8,16,24,32]:
                used=rows[:budget];best=max(float(singles.loc[damage,'W_display']),max(r['W_display'] for r in used))
                records.append(dict(damage=damage,method=method,budget=budget,repeat=0,best_W=best,success=best>=.5,
                    count_robust_success=max(r['count_ci_low'] for r in used)>=.5,actual_queries=len(used),complete=len(rows)==32,
                    first_success_batch_cost=next((r['batch_cost'] for r in used if r['W_display']>=.5),None)))
    assert len(sessions)==len(set(sessions))==len(p['damage_tasks'])
    runs=pd.DataFrame(records);runs.to_csv(out/'runs.csv',index=False);pd.DataFrame(queries).to_csv(out/'query_log.csv',index=False)
    (out/'decisions.json').write_text(json.dumps(decisions,indent=2)+'\n');(out/'cost_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    core=p.get('core_damage_tasks',p['damage_tasks'])
    matched=runs[runs.damage.isin(core)]
    if p['split']=='development':
        baseline=pd.read_csv(ROOT/'experiments/gb1_pilot_v1/runs.csv')
        baseline=baseline[(baseline.method=='wt_single_rank')&baseline.damage.isin(core)]
        legacy=pd.read_csv(ROOT/'experiments/gb1_agnes_v1/runs.csv');legacy=legacy[legacy.damage.isin(core)].copy();legacy['method']='Agnes_previous_menu'
        matched=pd.concat([matched,baseline,legacy],ignore_index=True)
    else:
        fixed=pd.read_csv(out/'fixed_policies/runs.csv')
        # Same seed0 and same task set for all actual-model comparisons.
        fixed=fixed[(fixed['repeat']==0)&fixed.budget.isin([8,16,24,32])].copy()
        fixed['method']=fixed.method.map({'wt_rank':'WT_rank',p['selected_method']:'fixed_full_pool'})
        matched=pd.concat([matched,fixed],ignore_index=True)
    summary=matched.groupby(['method','budget']).agg(success_rate=('success','mean'),count_robust_success_rate=('count_robust_success','mean'),mean_best_W=('best_W','mean'),tasks=('damage','nunique')).reset_index()
    summary.to_csv(out/'comparison.csv',index=False)
    if p.get('debug_damage_tasks'):
        runs[runs.damage.isin(p['debug_damage_tasks'])].to_csv(out/'debug_runs.csv',index=False)
    tokens={n:sum(e.get('tokens',{}).get(n,0) for e in ledger) for n in ['input','output','cacheRead','reasoning']}
    audit=dict(status='passed',tasks=len(sessions),core_tasks=len(core),complete_tasks=len(sessions)-len(warnings),warnings=warnings,
               actual_model_unique_queries=sum(r['method']=='Agnes_structure' for r in queries),model_batches=len(decisions),native_tool_calls=tool_calls,metadata_only_calls=metadata_calls,
               causal_menu_replay_batches=replay,ledger_inference_calls=len(ledger),tokens=tokens,credential_leak_matches=0,
               protocol_sha256=hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest(),split=p['split'],evidence='same public menu generator; each trajectory receives only its own prior feedback; old-menu comparison is descriptive, same-menu fixed is the attribution control')
    (out/'verification.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2));print(summary.query('budget==32').to_string(index=False))


if __name__=='__main__':main()
