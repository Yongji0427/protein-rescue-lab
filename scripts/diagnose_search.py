"""Post-run evaluator diagnosis; hidden-label outputs never enter search policies."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_policy import AdaptivePolicy
from run_pilot import count_interval


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/gb1_search_diagnosis_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tasks=pd.read_csv(ROOT/'experiments/gb1_pilot_v1/development_tasks.csv')
    singles=pd.read_csv(ROOT/'data/public_reference/singles.csv').set_index('mutant')
    protocol=json.loads((ROOT/'experiments/gb1_pilot_v1/protocol.json').read_text())
    raw=pd.read_csv(ROOT/'data/evaluator_only/gb1_raw.csv')
    held=protocol['holdout_full_sequence_positions']
    raw=raw.loc[raw['Mut2 Position'].notna() & ~(raw['Mut1 Position']+226).isin(held) & ~(raw['Mut2 Position']+226).isin(held)].copy()
    raw['ci_low'],_=count_interval(raw)
    parts=raw.mutant.str.split(':',expand=True);raw['r1'],raw['r2']=parts[0],parts[1]
    g1={k:g for k,g in raw.groupby('r1',sort=False)};g2={k:g for k,g in raw.groupby('r2',sort=False)};empty=raw.iloc[:0]
    agnes_decisions=json.loads((ROOT/'experiments/gb1_agnes_v1/decisions.json').read_text())
    menus={}
    for d in agnes_decisions:menus.setdefault(d['damage'],set()).update(x['id'] for x in d['menu']['candidates'])
    rows=[]
    for task in tasks.itertuples():
        frame=pd.concat([g1.get(task.damage,empty),g2.get(task.damage,empty)]).copy()
        frame['rescue']=np.where(frame.r1==task.damage,frame.r2,frame.r1)
        frame=frame.sort_values('rescue').reset_index(drop=True)
        ids=frame.rescue.to_numpy();wt=frame.rescue.map(singles.lnW).to_numpy()
        policy=AdaptivePolicy(ids,wt,float(singles.loc[task.damage,'lnW']),'contrast',0)
        warm=list(policy.order[:8])
        policy.update(warm[:4],frame.iloc[warm[:4]].lnW.tolist());policy.update(warm[4:],frame.iloc[warm[4:]].lnW.tolist())
        prediction=policy.predict()
        predicted_rank=np.empty(len(ids),int);predicted_rank[np.lexsort((ids,-prediction))]=np.arange(1,len(ids)+1)
        wt_rank=np.empty(len(ids),int);wt_rank[policy.order]=np.arange(1,len(ids)+1)
        for idx in frame.index[frame.ci_low>=.5]:
            row=frame.loc[idx]
            rows.append({'damage':task.damage,'rescue':row.rescue,'combination_W':float(np.exp(row.lnW)),
                'combo_ci_low':row.ci_low,'rescue_WT_W':float(singles.loc[row.rescue,'W']),
                'WT_rank':int(wt_rank[idx]),'contrast_rank_after8':int(predicted_rank[idx]),
                'in_first8':idx in warm,'in_Agnes_cohort':task.damage in menus,
                'appeared_in_Agnes_menu':row.rescue in menus.get(task.damage,set())})
    candidates=pd.DataFrame(rows);candidates.to_csv(OUT/'confirmed_rescue_ranks.csv',index=False)
    cohort=pd.read_csv(ROOT/'experiments/gb1_agnes_v1/cohort_audit.csv')
    original=pd.read_csv(ROOT/'experiments/gb1_pilot_v1/runs.csv').query("method=='wt_single_rank'")
    covered8=int(original.query('budget==8').count_robust_success.sum())
    covered32=int(original.query('budget==32').count_robust_success.sum())
    diagnostics={'development_tasks':len(tasks),'development_confirmed_reachable':int((tasks.robust_rescue_candidates>0).sum()),
        'WT_confirmed_found8':covered8,'WT_confirmed_found32':covered32,
        'Agnes_cohort_confirmed_reachable':int((cohort.robust_rescue_candidates>0).sum()),
        'Agnes_cohort_confirmed_found32':8,
        'confirmed_sign_reversal_backgrounds':int(candidates.loc[candidates.rescue_WT_W<1,'damage'].nunique()),
        'confirmed_sign_reversal_menu_coverage':candidates.loc[(candidates.rescue_WT_W<1)&candidates.in_Agnes_cohort,['damage','rescue','appeared_in_Agnes_menu']].to_dict('records'),
        'evidence_boundary':'post-run development diagnosis; not accessible to agent; no heldout labels analyzed'}
    (OUT/'diagnosis.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print(json.dumps(diagnostics,indent=2))
    print(candidates.query("damage in ['V265G','V265W']").to_string(index=False))


if __name__=='__main__':main()
