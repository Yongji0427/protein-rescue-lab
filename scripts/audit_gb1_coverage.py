"""Post-test GB1 coverage audit; never reuse expanded scores as independent validation."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from run_pilot import count_interval
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/dataset_feasibility_v1'


def main():
    p=json.loads((ROOT/'experiments/gb1_pilot_v1/protocol.json').read_text());held=p['holdout_full_sequence_positions']
    raw=pd.read_csv(ROOT/'data/evaluator_only/gb1_raw.csv');singles=pd.read_csv(ROOT/'data/public_reference/singles.csv').set_index('mutant')
    damage=singles[singles.W_display<=.2];dev_damage=damage[~damage.p1.isin(held)]
    doubles=raw[raw['Mut2 Position'].notna()].copy();a=doubles['Mut1 Position']+226;b=doubles['Mut2 Position']+226
    counts={'raw_records':len(raw),'single_records':len(raw)-len(doubles),'double_records':len(doubles),'damage_singles_total':len(damage),'damage_development':len(dev_damage),'damage_reserved':len(damage)-len(dev_damage),'development_double_records':int((~a.isin(held)&~b.isin(held)).sum()),'cross_split_double_records':int((a.isin(held)^b.isin(held)).sum()),'both_reserved_double_records':int((a.isin(held)&b.isin(held)).sum()),'excluded_damage_tasks':json.loads((ROOT/'experiments/gb1_pilot_v1/excluded_tasks.json').read_text())}
    # Only development D is audited. Both-reserved combinations are never consumed.
    parts=doubles.mutant.str.split(':',expand=True);doubles['r1'],doubles['r2']=parts[0],parts[1]
    relevant=doubles.r1.isin(dev_damage.index)|doubles.r2.isin(dev_damage.index)
    doubles=doubles[relevant].copy();doubles['ci_low'],_=count_interval(doubles);doubles['W_display']=np.exp(doubles.lnW)
    groups1={k:g for k,g in doubles.groupby('r1',sort=False)};groups2={k:g for k,g in doubles.groupby('r2',sort=False)};empty=doubles.iloc[:0]
    rows=[]
    for d in dev_damage.index:
        frame=pd.concat([groups1.get(d,empty),groups2.get(d,empty)]).copy();frame['rescue']=np.where(frame.r1==d,frame.r2,frame.r1)
        for pool in ['original_development','expanded_all_R','expanded_harmful_R']:
            f=frame.copy()
            if pool=='original_development':f=f[~f.rescue.str[1:-1].astype(int).isin(held)]
            if pool=='expanded_harmful_R':f=f[f.rescue.map(singles.count_ci_high)<1]
            f['single_lnW']=f.rescue.map(singles.lnW);f=f.sort_values(['single_lnW','rescue'],ascending=[False,True])
            record=dict(damage=d,pool=pool,candidates=len(f),point_reachable=bool((f.W_display>=.5).any()),robust_reachable=bool((f.ci_low>=.5).any()))
            for k in [8,32]:record[f'WT_point_{k}']=bool((f.head(k).W_display>=.5).any());record[f'WT_robust_{k}']=bool((f.head(k).ci_low>=.5).any())
            rows.append(record)
    tasks=pd.DataFrame(rows);tasks.to_csv(OUT/'gb1_coverage_tasks.csv',index=False)
    summary=tasks.groupby('pool').agg(tasks=('damage','size'),candidate_min=('candidates','min'),candidate_median=('candidates','median'),candidate_max=('candidates','max'),point_reachable=('point_reachable','sum'),robust_reachable=('robust_reachable','sum'),WT_point8=('WT_point_8','sum'),WT_point32=('WT_point_32','sum'),WT_robust8=('WT_robust_8','sum'),WT_robust32=('WT_robust_32','sum')).reset_index()
    summary.to_csv(OUT/'gb1_coverage_summary.csv',index=False)
    counts.update(evidence='post-test expanded development-D audit; additional cross-split R labels disclosed only in this audit; original frozen results unchanged; not a new independent test',both_reserved_double_labels_analyzed=0)
    (OUT/'gb1_coverage_audit.json').write_text(json.dumps(counts,indent=2)+'\n');print(json.dumps(counts,indent=2));print(summary.to_string(index=False))
    old=pd.read_csv(ROOT/'experiments/gb1_pilot_v1/development_tasks.csv').set_index('damage')
    f=tasks[tasks.pool=='original_development'].set_index('damage')
    assert np.array_equal(f.loc[old.index,'candidates'],old.candidates)
    assert np.array_equal(f.loc[old.index,'point_reachable'],old.rescue_candidates>0)
    assert np.array_equal(f.loc[old.index,'robust_reachable'],old.robust_rescue_candidates>0)


if __name__=='__main__':main()
