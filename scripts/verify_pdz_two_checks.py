"""Rebuild counting evidence, conditional score arithmetic and ranking coverage."""
import os
os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
import gzip,json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import beta
from pdz2018_data import ROOT,AA,DOMAINS,RT,dump,load_development
from pdz2018_uncertainty import TAIL_ALPHA
from pdz_two_checks import OUT,UNC,METHODS,MODEL,check,contexts,Scores


def main():
    check();assert json.loads((OUT/'execution.json').read_text())['status']=='completed'
    raw=loadmat(ROOT/'data/pdz2018/source/Salinas_Ranganathan_data.mat',variable_names=list(DOMAINS),squeeze_me=True,struct_as_record=False)
    assert 'shank3' not in raw and 'zo1' not in raw
    cal=pd.read_csv(ROOT/'experiments/dataset_feasibility_v1/pdz2018_calibration_QC.csv').set_index('domain');db=pd.read_csv(UNC/'double_bounds.csv');sb=pd.read_csv(UNC/'single_bounds.csv')
    rebuilt={};full={};checked=0;checked_singles=0
    for domain in DOMAINS:
        o=raw[domain];a=db[db.domain==domain];b=sb[sb.domain==domain]
        if domain=='pdz3':assert a.counting_lower.isna().all() and b.counting_upper.isna().all();continue
        alpha,fw=cal.loc[domain,['alpha','fraction_WT']]
        def direct(s,n):
            s=np.asarray(s);n=np.asarray(n);ql=np.zeros(len(s));sel=s>0;ql[sel]=beta.ppf(TAIL_ALPHA,s[sel],n[sel]+1)
            qu=beta.ppf(1-TAIL_ALPHA,s+1,n)
            wl=beta.ppf(TAIL_ALPHA,o.selected_wt_counts,o.unselected_wt_counts+1);wu=beta.ppf(1-TAIL_ALPHA,o.selected_wt_counts+1,o.unselected_wt_counts)
            ratios=[(ql/(1-ql))/(wu/(1-wu)),(qu/(1-qu))/(wl/(1-wl))];out=[]
            for r in ratios:
                frac=fw*r**alpha;y=np.full(len(r),np.inf);valid=frac<1;y[valid]=frac[valid]*(1-fw)/(fw*(1-frac[valid]));out.append(y)
            return out
        idx=[(int(r.mutation1[1:-1])-1,int(r.mutation2[1:-1])-1,AA.index(r.mutation1[-1]),AA.index(r.mutation2[-1])) for r in a.itertuples()]
        lo,hi=direct([o.selected_double_counts[k] for k in idx],[o.unselected_double_counts[k] for k in idx]);assert np.allclose(lo,a.counting_lower,atol=1e-12,rtol=1e-10) and np.allclose(hi,a.counting_upper,rtol=1e-10)
        point=np.array([np.exp(-o.deltaG_double[k]/RT) for k in idx]);assert np.allclose(point,a.point_affinity)
        for r,support in zip(a.itertuples(),(lo>=.5)&(point>=.5)):
            rebuilt[domain,r.mutation1,r.mutation2]=rebuilt[domain,r.mutation2,r.mutation1]=bool(support)
        idx=[(AA.index(r.mutation[-1]),int(r.mutation[1:-1])-1) for r in b.itertuples()]
        lo,hi=direct([o.selected_single_counts[k] for k in idx],[o.unselected_single_counts[k] for k in idx]);assert np.allclose(lo,b.counting_lower,atol=1e-12,rtol=1e-10) and np.allclose(hi,b.counting_upper,rtol=1e-10)
        for r,h in zip(b.itertuples(),hi):full[domain,r.mutation]=bool(h<=.2)
        checked+=len(a);checked_singles+=len(b)
    tasks=load_development();res=pd.read_csv(OUT/'rescored_trajectories.csv').set_index(['run','scheme','method','task_id']);assert res.index.is_unique
    trajectory_count=0
    for folder in res.index.get_level_values('run').unique():
        with gzip.open(ROOT/'experiments'/folder/'trajectories.jsonl.gz','rt') as stream:
            for line in stream:
                r=json.loads(line);t=tasks[r['task_id']];q=sum(r['batches'],[]);row=res.loc[folder,r['scheme'],r['method'],r['task_id']]
                values={v['id']:v['affinity'] for v in t['rows']};assert row.point_success==any(values[m]>=.5 for m in q)
                if t['domain']=='pdz3':assert pd.isna(row.count_supported) and not row.count_available
                else:
                    yes=any(rebuilt[t['domain'],t['damage'],m] for m in q)
                    assert bool(row.count_supported)==yes and bool(row.full_count_supported)==bool(yes and full[t['domain'],t['damage']])
                trajectory_count+=1
    assert trajectory_count==len(res)
    # Independent CPU inference on WT and mutant backgrounds, including both directions.
    import torch
    from transformers import AutoTokenizer,EsmForMaskedLM
    torch.set_num_threads(2);tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);model=EsmForMaskedLM.from_pretrained(MODEL,local_files_only=True).eval()
    vocab=[tokenizer.convert_tokens_to_ids(a) for a in AA];store=Scores();cpu_checks=[]
    for context in json.loads((ROOT/'experiments/pdz2018_esm_v2/sequence_mapping.json').read_text()):
        inputs=list(contexts(context));cached=store.data[context['domain']];assert set(cached)==set(inputs) and len(inputs)==1377
        for damage,j in [inputs[0],inputs[9],inputs[-1]]:
            seq=list(context['sequence']);offset=context['helix_start_0based']
            if damage!='WT':seq[offset+int(damage[1:-1])-1]=damage[-1]
            encoded=tokenizer(''.join(seq),return_tensors='pt');encoded['input_ids'][0,offset+j+1]=tokenizer.mask_token_id
            with torch.inference_mode():p=model(**encoded).logits[0,offset+j+1].log_softmax(-1)[vocab].numpy()
            error=float(abs(p-cached[damage,j]).max());assert np.allclose(p,cached[damage,j],atol=.005,rtol=.001),error
            cpu_checks.append(dict(domain=context['domain'],damage=damage,target=j,max_abs_error=error))
    coupling_checks=[]
    for context in json.loads((ROOT/'experiments/pdz2018_esm_v2/sequence_mapping.json').read_text()):
        helix=context['helix'];domain=context['domain'];d=helix[0]+'1'+next(a for a in AA if a!=helix[0]);r=helix[1]+'2'+next(a for a in AA if a!=helix[1]);od=context['helix_start_0based']
        values={}
        for bg,target in [('WT',0),('WT',1),(d,1),(r,0)]:
            seq=list(context['sequence'])
            if bg!='WT':seq[od+int(bg[1:-1])-1]=bg[-1]
            tokens=tokenizer(''.join(seq),return_tensors='pt');tokens['input_ids'][0,od+target+1]=tokenizer.mask_token_id
            with torch.inference_mode(): values[bg,target]=model(**tokens).logits[0,od+target+1].log_softmax(-1)[vocab].numpy()
        def ll(bg,m):
            v=values[bg,int(m[1:-1])-1];return float(v[AA.index(m[-1])]-v[AA.index(m[0])])
        actual=.5*(ll(d,r)-ll('WT',r)+ll(r,d)-ll('WT',d));cached=store.coupling(domain,d,r)
        assert abs(actual-cached)<.001
        coupling_checks.append(dict(domain=domain,damage=d,rescue=r,cpu=actual,cached=cached,absolute_error=abs(actual-cached)))
    result=pd.read_csv(OUT/'signal_tasks.csv').set_index(['task_id','method']);rankings=pd.read_csv(OUT/'signal_rankings.csv.gz');groups={(tid,m):g for (tid,m),g in rankings.groupby(['task_id','method'],sort=False)}
    assert len(result)==207*len(METHODS) and result.index.is_unique
    symmetry=0
    for tid,t in tasks.items():
        ids=[r['id'] for r in t['rows']];base=store.task(t);order=sorted(range(len(ids)),key=lambda i:(-base['wt_single'][i],ids[i]));initial=order[:8];remaining=order[8:]
        for r in t['rows']:
            assert abs(store.coupling(t['domain'],t['damage'],r['id'])-store.coupling(t['domain'],r['id'],t['damage']))<1e-12;symmetry+=1
        for method in METHODS:
            # Verify saved rankings independently, including the fixed rank-fusion formula.
            if method=='wt_coupling_rank':
                from scipy.stats import rankdata
                scores=-rankdata(-base['wt_single'][remaining])-rankdata(-base['coupling_sym'][remaining]);mapping=dict(zip(remaining,scores));rank=sorted(remaining,key=lambda i:(-mapping[i],ids[i]))
            else:rank=sorted(remaining,key=lambda i:(-base[method][i],ids[i]))
            actual=groups[tid,method];assert list(actual.rescue)==[ids[i] for i in rank]
            row=result.loc[tid,method]
            for k in [16,24,32]:assert bool(row[f'potential{k}'])==any(t['rows'][i]['affinity']>=.5 for i in (initial+rank[:max(0,min(k,len(ids))-8)]))
            if t['domain']=='pdz3':assert pd.isna(row.count_supported)
            else:
                chosen=initial+rank[:min(32,len(ids))-8];yes=any(rebuilt[t['domain'],t['damage'],ids[i]] for i in chosen)
                assert bool(row.count_supported)==yes and bool(row.full_count_supported)==bool(yes and full[t['domain'],t['damage']])
    # The WT static reference must reproduce the original full32 point outcome.
    old=pd.read_csv(ROOT/'experiments/pdz2018_portfolio_v1/runs.csv').query("scheme=='site' and method=='wt_rank'").set_index('task_id')
    for tid,row in old.iterrows():assert bool(result.loc[tid,'wt_single'].potential32)==bool(row.success32)
    check();dump(OUT/'verification.json',dict(status='passed',double_bounds_rebuilt=checked,single_bounds_rebuilt=checked_singles,
        saved_trajectories_rescored=trajectory_count,masked_contexts_audited=4131,cpu_logprob_checks=cpu_checks,cpu_coupling_checks=coupling_checks,
        symmetric_pair_checks=symmetry,static_task_method_rankings_checked=len(result),WT_reference_exact=True,
        missing_PDZ3_preserved=True,old_artifacts_unchanged=True,new_adaptive_queries=0,new_supervised_fits=0,new_agnes_calls=0,sealed_domains_loaded=False))
    print('Two checks independently verified',flush=True)


if __name__=='__main__':main()
