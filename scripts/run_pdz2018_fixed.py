"""Full 207-task development evaluation with one frozen policy transfer."""
import gzip
import csv
import json
import numpy as np
import pandas as pd
from pdz2018_data import ROOT, OUT, dump, load_development
from pdz2018_policy import PDZPolicy


def main():
    assert not (OUT/'fixed_runs.csv').exists()
    p=json.loads((OUT/'protocol.json').read_text());tasks=load_development();records=[]
    with gzip.open(OUT/'fixed_query_log.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['task_id','method','repeat','query','batch','rescue','affinity','count_lower_affinity'])
        writer.writeheader()
        for task_id,t in tasks.items():
            for method in p['methods']:
                for repeat in range(1 if method=='wt_rank' else p['repeats']):
                    domain=list(dict.fromkeys(tasks[k]['domain'] for k in tasks)).index(t['domain'])
                    seed=np.random.SeedSequence([p['seed'],domain,int(t['damage'][1:-1]),ord(t['damage'][-1]),repeat])
                    policy=PDZPolicy(t,method,seed);best=0.;best_low=0.;first=None
                    batch=0
                    while len(policy.seen)<policy.cap:
                        chosen=policy.proposal();ids=[str(policy.ids[i]) for i in chosen];batch+=1
                        for i in chosen:
                            row=t['rows'][i];best=max(best,row['affinity'])
                            if row['count_lower_affinity'] is not None:best_low=max(best_low,row['count_lower_affinity'])
                            writer.writerow(dict(task_id=task_id,method=method,repeat=repeat,query=len(policy.seen)+ids.index(row['id'])+1,
                                batch=batch,rescue=row['id'],affinity=row['affinity'],count_lower_affinity=row['count_lower_affinity']))
                        policy.update(ids)
                        if first is None and best>=.5:first=len(policy.seen)
                    records.append(dict(task_id=task_id,domain=t['domain'],damage=t['damage'],method=method,repeat=repeat,
                        queries_used=len(policy.seen),success=best>=.5,best_affinity=best,first_success_batch_cost=first,
                        count_supported_success=best_low>=.5 if t['count_support_available'] else None))
    frame=pd.DataFrame(records);frame.to_csv(OUT/'fixed_runs.csv',index=False)
    summary=frame.groupby(['domain','method']).agg(success_rate=('success','mean'),tasks=('task_id','nunique'),runs=('task_id','size')).reset_index()
    summary.to_csv(OUT/'fixed_summary.csv',index=False)
    dump(OUT/'fixed_execution.json',dict(tasks=len(tasks),runs=len(frame),queries=int(frame.queries_used.sum()),
        model_calls=0,unread_test_domains=['shank3','zo1'],endpoint='point-only development'))
    print(summary.to_string(index=False))


if __name__=='__main__':main()
