"""Run a frozen full development factorial or held-out fixed-policy evaluation."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from experiment_data import ROOT, load_frames
from factor_policy import FactorPolicy
from run_pilot import QueryOracle


def main():
    out = ROOT / sys.argv[1]
    p = json.loads((out / 'protocol.json').read_text())
    assert not (out / 'runs.csv').exists(), 'Do not overwrite a completed experiment'
    start = time.perf_counter()
    singles, frames = load_frames(p, out)
    records = []
    fields = ['damage','method','repeat','query','batch_cost','rescue','slot','lnW','W_display','count_ci_low','wt_rescue_lnW','predicted_lnW_before_batch']
    with gzip.open(out / 'query_log.csv.gz', 'wt', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for damage, frame in frames.items():
            ids = frame.rescue.to_numpy(); wt = singles.loc[ids, 'lnW'].to_numpy()
            dy = float(singles.loc[damage, 'lnW'])
            labels = {r.rescue: r for r in frame.itertuples()}
            for method in p['methods']:
                template = FactorPolicy(ids, wt, dy, damage, method, 0)
                repeats = 1 if method == 'wt_rank' else p['repeats']
                for repeat in range(repeats):
                    seed = np.random.SeedSequence([p['seed'],int(damage[1:-1]),ord(damage[-1]),repeat])
                    policy = template.replicate(seed)
                    oracle = QueryOracle(damage, labels, 32)
                    best, robust, first, first_r = float(np.exp(dy)), 0., None, None
                    for cost in range(4,33,4):
                        prediction = policy.predict(); batch = policy.choose_batch()
                        indices, feedback = [], []
                        for i, slot in batch:
                            row = oracle.query(str(ids[i])); indices.append(i); feedback.append(row.lnW)
                            best = max(best,row.W_display); robust = max(robust,row.ci_low)
                            if first is None and row.W_display >= .5: first = cost
                            if first_r is None and row.ci_low >= .5: first_r = cost
                            writer.writerow(dict(zip(fields,[damage,method,repeat,oracle.used,cost,ids[i],slot,row.lnW,row.W_display,row.ci_low,wt[i],prediction[i]])))
                        policy.update(indices,feedback)
                        assert oracle.used == cost
                        records.append(dict(damage=damage,full_position=int(damage[1:-1]),method=method,repeat=repeat,budget=cost,best_W=best,success=best>=.5,count_robust_success=robust>=.5,strong_success=best>=1,first_success_batch_cost=first,first_robust_success_batch_cost=first_r,queries_used=oracle.used))
            print('Completed '+damage,flush=True)
    runs = pd.DataFrame(records); runs.to_csv(out/'runs.csv',index=False)
    summary = runs.groupby(['method','budget']).agg(success_rate=('success','mean'),count_robust_success_rate=('count_robust_success','mean'),mean_best_W=('best_W','mean'),strong_success_rate=('strong_success','mean')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    per = runs.groupby(['damage','method','budget'])[['success','count_robust_success','best_W']].mean().reset_index()
    per.to_csv(out/'per_task.csv',index=False)
    audit = dict(protocol_sha256=hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest(),tasks=len(frames),split=p['split'],queries=int(runs.query('budget==32').queries_used.sum()),repeats=p['repeats'],wall_seconds=time.perf_counter()-start)
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(summary.query('budget==32').to_string(index=False))


if __name__ == '__main__': main()
