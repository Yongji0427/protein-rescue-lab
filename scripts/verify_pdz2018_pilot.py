"""Independently validate source labels, budgets and recorded search decisions."""
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from pdz2018_data import ROOT,OUT,DOMAINS,AA,RT,dump,load_development
from pdz2018_policy import PDZPolicy


def main():
    p=json.loads((OUT/'protocol.json').read_text());freeze=json.loads((OUT/'freeze.json').read_text())
    addendum=json.loads((OUT/'setup_addendum.json').read_text())
    effective={**freeze['hashes'],**addendum['hashes']}
    assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest for path,digest in effective.items())
    gb1=json.loads((ROOT/'experiments/gb1_final_holdout_v1/freeze.json').read_text())
    assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest for path,digest in gb1['hashes'].items())
    source=loadmat(ROOT/'data/pdz2018/source/Salinas_Ranganathan_data.mat',variable_names=list(DOMAINS),squeeze_me=True,struct_as_record=False)
    assert 'shank3' not in source and 'zo1' not in source
    def measurement(task_id,rescue):
        domain,damage=task_id.split('__');o=source[domain]
        i,a=int(damage[1:-1])-1,AA.index(damage[-1]);j,b=int(rescue[1:-1])-1,AA.index(rescue[-1])
        assert i!=j and damage[0]==DOMAINS[domain][0][i] and rescue[0]==DOMAINS[domain][0][j]
        if i>j:i,j,a,b=j,i,b,a
        return float(np.exp(-o.deltaG_double[i,j,a,b]/RT))
    tasks=load_development();runs=pd.read_csv(OUT/'fixed_runs.csv')
    assert set(runs.task_id)==set(p['all_damage_tasks'])==set(tasks)
    queries=0;groups={};seed0={}
    for chunk in pd.read_csv(OUT/'fixed_query_log.csv.gz',chunksize=100000):
        values={key:measurement(*key) for key in set(zip(chunk.task_id,chunk.rescue))}
        expected=np.array([values[k] for k in zip(chunk.task_id,chunk.rescue)])
        assert np.allclose(expected,chunk.affinity,rtol=1e-12,atol=1e-12)
        for (task_id,method,repeat),f in chunk.groupby(['task_id','method','repeat'],sort=False):
            key=(task_id,method,int(repeat));current=groups.setdefault(key,dict(ids=[],success=False,best=0.))
            assert f['query'].tolist()==list(range(len(current['ids'])+1,len(current['ids'])+1+len(f)))
            assert not set(f.rescue)&set(current['ids']) and not f.rescue.duplicated().any()
            current['ids']+=f.rescue.tolist();current['success']|=bool((f.affinity>=.5).any());current['best']=max(current['best'],float(f.affinity.max()))
            if repeat==0:seed0.setdefault((task_id,method),[]).extend(f.rescue.tolist())
        queries+=len(chunk)
    for r in runs.itertuples():
        g=groups[r.task_id,r.method,r.repeat]
        assert len(g['ids'])==r.queries_used==min(32,len(tasks[r.task_id]['rows']))
        assert g['success']==r.success and np.isclose(g['best'],r.best_affinity)
    # Replay every seed-zero decision and every model menu before each feedback.
    replayed=0
    for task_id,t in tasks.items():
        domain=list(DOMAINS).index(t['domain']);seed=np.random.SeedSequence([p['seed'],domain,int(t['damage'][1:-1]),ord(t['damage'][-1]),0])
        for method in p['methods']:
            policy=PDZPolicy(t,method,seed);ids=seed0[task_id,method];start=0
            while start<len(ids):
                proposal=[str(policy.ids[i]) for i in policy.proposal()]
                assert proposal==ids[start:start+len(proposal)]
                policy.update(proposal);start+=len(proposal);replayed+=1
    model_queries=0;complete=0;deviations=[];corrections=[];runtime_errors=[];native_calls=0;inferences=0;leaks=0
    from agnes_client import read_config
    credential=read_config()['AGNES_API_KEY']
    for task_id in p['damage_tasks']:
        path=OUT/'queried_results'/(task_id+'.json');s=json.loads(path.read_text());t=tasks[task_id]
        domain=list(DOMAINS).index(t['domain']);seed=np.random.SeedSequence([p['seed'],domain,int(t['damage'][1:-1]),ord(t['damage'][-1]),0])
        policy=PDZPolicy(t,'no_feedback_contact',seed);menu=None
        assert len(s['queries'])<=32 and len({r['rescue'] for r in s['queries']})==len(s['queries'])
        for decision in s['decisions']:
            assert decision['budget_before']==len(policy.seen)
            ids=decision['selected'];assert len(ids)==4 and len(set(ids))==4
            if len(policy.seen)<8:
                assert ids==[str(policy.ids[i]) for i in policy.proposal()]
            else:
                assert menu==decision['menu'] and menu['required_WT_anchor'] in ids
                assert set(ids)<={r['id'] for r in menu['candidates']}
            policy.update(ids);menu=policy.menu();replayed+=1
        assert [r['rescue'] for r in s['queries']]==[x for d in s['decisions'] for x in d['selected']]
        for r in s['queries']:
            assert np.isclose(r['affinity'],measurement(task_id,r['rescue']),rtol=1e-12,atol=1e-12)
            if t['domain']=='pdz3':assert r['count_lower_affinity'] is None
        result=json.loads((OUT/'agh_events'/(task_id+'.run.json')).read_text())
        assert result['queries']==len(s['queries']);complete+=len(s['queries'])==32;model_queries+=len(s['queries'])
        if len(result['turns'])>1:corrections.append(dict(task_id=task_id,turns=len(result['turns']),queries=len(s['queries'])))
        for turn in result['turns']:
            if isinstance(turn.get('outcome'),dict) and turn['outcome'].get('error'):
                runtime_errors.append(dict(task_id=task_id,attempt=turn['attempt'],before=turn['before'],after=turn['after'],error=turn['outcome']['error']))
        trace=(OUT/'agh_events'/(task_id+'.jsonl')).read_text()
        leaks+=credential in trace
        events=[json.loads(line) for line in trace.splitlines()]
        assert all(e['ts']>=addendum['frozen_at_utc'] for e in events if e['type']=='request/header')
        calls=[e['data'] for e in events if e['type']=='tool/call'];native_calls+=len(calls)
        inferences+=sum(e['type']=='request/header' for e in events)
        allowed_calls=[]
        for call in calls:
            name=call['name']
            if name=='query_pdz_batch':
                assert call['args']['task_token']==json.loads((OUT/'public_tasks'/(task_id+'.json')).read_text())['task_token']
                allowed_calls.append(call['args']['rescue_ids'])
            else:
                metadata=name=='tool_describe' and call['args']=={'name':'query_pdz_batch'}
                missing=any(e['type']=='tool/result' and e['data'].get('toolUseId')==call['toolUseId'] and e['data'].get('code')=='TOOL_NOT_FOUND' for e in events)
                assert metadata or missing or name=='todo', 'Unexpected unrelated data-access tool: '+name
                deviations.append(dict(task_id=task_id,name=name,scope='schema metadata' if metadata else 'rejected unknown tool' if missing else 'internal todo'))
        observed_batches=[d['selected'] for d in s['decisions'][2:]]
        # Accepted calls occur in order; rejected query calls may be interleaved.
        cursor=0
        for ids in allowed_calls:
            if cursor<len(observed_batches) and ids==observed_batches[cursor]:cursor+=1
        assert cursor==len(observed_batches)
    assert leaks==0
    dump(OUT/'verification.json',dict(status='passed',fixed_tasks=len(tasks),fixed_runs=len(runs),fixed_queries=queries,
        model_tasks=len(p['damage_tasks']),complete_model_tasks=complete,model_queries=model_queries,
        decision_batches_replayed=replayed,old_GB1_hashes_unchanged=len(gb1['hashes']),
        effective_frozen_files_unchanged=len(effective),sealed_parent_arrays_loaded=False,credential_leak_count=leaks,
        engineering_complete=complete==32,native_tool_calls=native_calls,inference_requests=inferences,
        accepted_model_batches=(model_queries-8*len(p['damage_tasks']))//4,
        completion_corrections=corrections,runtime_errors=runtime_errors,trace_tool_deviations=deviations,
        protocol_conformity='tool scope deviations disclosed' if deviations else 'passed',
        claims='point-only development; uncertainty unavailable for PDZ3; no independent heldout improvement'))
    print('Verified:',queries,'fixed queries;',model_queries,'model queries;',complete,'complete model tasks')


if __name__=='__main__':main()
