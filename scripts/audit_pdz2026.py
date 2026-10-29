"""Actual experimental fitness audit, development assays only; test labels untouched."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/dataset_feasibility_v1';PUB=ROOT/'data/pdz2026/public';Z=norm.ppf(.95)


def main():
    calibration=pd.read_csv(OUT/'pdz2026_public_calibration.csv').set_index('source')
    cover=json.loads((OUT/'pdz2026_identity_coverage.json').read_text());tasks=[];examples=[];reads=[]
    for info in cover:
        if info['split']!='development':continue
        assay=info['assay'];s=pd.read_csv(PUB/(assay+'_singles.csv')).set_index('mutant');d=pd.read_csv(PUB/(assay+'_double_identity.csv'))
        d['activity']=np.nan;d['activity_sigma']=np.nan
        for source,indices in d.groupby('source').groups.items():
            assert 'erbin' not in source
            raw=pd.read_csv(ROOT/source,usecols=['fitness','sigma'])
            rowids=d.loc[indices,'row_index'].to_numpy(int);c=calibration.loc[source]
            d.loc[indices,'activity']=(raw.loc[rowids,'fitness'].to_numpy()-c.stop_single_median)/c.dynamic_range
            d.loc[indices,'activity_sigma']=raw.loc[rowids,'sigma'].to_numpy()/c.dynamic_range
            reads.append(dict(source=source,queried_for_audit=len(indices)))
        valid=np.isfinite(d.activity)&np.isfinite(d.activity_sigma)&(d.activity_sigma>=0)
        d=d[valid].copy();d['lower']=d.activity-Z*d.activity_sigma
        s=s[np.isfinite(s.activity)&np.isfinite(s.activity_sigma)&(s.activity_sigma>=0)].copy();s['upper']=s.activity+Z*s.activity_sigma
        parts=d.mutant.str.split(':',expand=True);d['a'],d['b']=parts[0],parts[1]
        g1={key:frame for key,frame in d.groupby('a',sort=False)};g2={key:frame for key,frame in d.groupby('b',sort=False)};empty=d.iloc[:0]
        damaged=s[s.activity<=.2]
        for damage,row in damaged.iterrows():
            f=pd.concat([g1.get(damage,empty),g2.get(damage,empty)]).copy();f['rescue']=np.where(f.a==damage,f.b,f.a);f=f[f.rescue.isin(s.index)]
            f['R_activity']=f.rescue.map(s.activity);f['R_upper']=f.rescue.map(s.upper)
            for regime in ['all_R','point_harmful_R','confident_harmful_R']:
                frame=f.copy()
                if regime=='point_harmful_R':frame=frame[frame.R_activity<1]
                if regime=='confident_harmful_R':frame=frame[frame.R_upper<1]
                frame=frame.sort_values(['R_activity','rescue'],ascending=[False,True])
                record=dict(assay=assay,domain='_'.join(assay.split('_')[:2]),damage=damage,damage_position=int(damage[1:-1]),regime=regime,candidates=len(frame),searchable=len(frame)>32,point_reachable=bool((frame.activity>=.5).any()),interval_supported_reachable=bool((frame.lower>=.5).any()))
                for k in [8,32]:record[f'WT_point{k}']=bool((frame.head(k).activity>=.5).any());record[f'WT_supported{k}']=bool((frame.head(k).lower>=.5).any())
                tasks.append(record)
                good=frame[(frame.lower>=.5)&(frame.R_upper<1)]
                if len(good):
                    best=good.sort_values('activity',ascending=False).iloc[0]
                    examples.append(dict(assay=assay,damage=damage,rescue=best.rescue,regime=regime,searchable=record['searchable'],D_activity=row.activity,R_activity=best.R_activity,combo_activity=best.activity,combo_lower=best.lower,source='development evaluator audit, not model discovery'))
    t=pd.DataFrame(tasks);t.to_csv(OUT/'pdz2026_tasks.csv',index=False);pd.DataFrame(examples).to_csv(OUT/'pdz2026_development_examples.csv',index=False)
    rows=[]
    for stratum in ['all_damage','searchable_gt32']:
        subset=t if stratum=='all_damage' else t[t.searchable]
        for (assay,regime),f in subset.groupby(['assay','regime']):
            rows.append(dict(stratum=stratum,assay=assay,regime=regime,tasks=len(f),candidate_min=int(f.candidates.min()),candidate_median=float(f.candidates.median()),candidate_max=int(f.candidates.max()),point_reachable=int(f.point_reachable.sum()),interval_supported_reachable=int(f.interval_supported_reachable.sum()),WT_point8=int(f.WT_point8.sum()),WT_point32=int(f.WT_point32.sum()),WT_supported8=int(f.WT_supported8.sum()),WT_supported32=int(f.WT_supported32.sum())))
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'pdz2026_summary.csv',index=False)
    # Deduplicate the same damage background across ligands within its domain.
    unique=t[t.searchable&(t.regime=='confident_harmful_R')].groupby(['domain','damage','damage_position']).agg(reachable=('interval_supported_reachable','max'),WT_any_ligand_found32=('WT_supported32','max')).reset_index()
    unique.to_csv(OUT/'pdz2026_unique_damage.csv',index=False)
    audit=dict(status='completed',development_binding_assays=6,test_binding_assays_unread=['erbin_pdz1_cript'],test_double_outcomes_analyzed=0,new_model_runs=0,representative_rule='largest summed input count before selection; AA duplicates not averaged as independent replicates',normalization='fixed WT-like synonymous and single-stop controls, per block; PCA selection proxy',interval='provided DiMSum sigma and normal90%, fixed normalization; not independent biological CI or literal binding fraction',development_sources=reads)
    (OUT/'pdz2026_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(summary[(summary.stratum=='searchable_gt32')&(summary.regime=='confident_harmful_R')].to_string(index=False));print('Unique reachable searchable backgrounds:',int(unique.reachable.sum()))


if __name__=='__main__':main()
