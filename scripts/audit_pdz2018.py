"""Development-only PDZ2018 feasibility; sealed domains are never loaded."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import least_squares
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/dataset_feasibility_v1';AA='ARNDCQEGHILKMFPSTWYV';RT=.00198720425864083*298.15


def low_ratio(selected,inputs,wt_ratio):
    positive=selected>0;lo=np.zeros_like(selected,dtype=float);lo[positive]=beta.ppf(.05,selected[positive],inputs[positive]+1)
    return lo/(1-lo)/wt_ratio


def high_ratio(selected,inputs,wt_ratio):
    hi=beta.ppf(.95,selected+1,inputs)
    return hi/(1-hi)/wt_ratio


def fit_calibration(o):
    wt=o.selected_wt_counts/o.unselected_wt_counts
    x=np.log(o.selected_single_counts/o.unselected_single_counts/wt);y=o.deltaG_single;mask=np.isfinite(x)&np.isfinite(y);x=x[mask];y=y[mask]
    def energy(raw,t):
        alpha,f=t;fm=np.clip(f*np.exp(alpha*raw),1e-10,1-1e-10)
        return RT*(np.log((1-fm)/fm)-np.log((1-f)/f))
    fit=least_squares(lambda t:energy(x,t)-y,[.5,.1],bounds=([.001,.00001],[10,.999]))
    return fit.x,float(np.sqrt(np.mean((energy(x,fit.x)-y)**2))),energy


def main():
    path=ROOT/'data/pdz2018/source/Salinas_Ranganathan_data.mat'
    names=['pdz2','pdz3','syntrophin'];source=loadmat(path,variable_names=names,squeeze_me=True,struct_as_record=False)
    assert 'shank3' not in source and 'zo1' not in source
    records=[];cover=[];calibration=[];examples=[]
    for name in names:
        o=source[name];native=[]
        for j in range(9):
            row=np.flatnonzero(~np.isfinite(o.unselected_single_counts[:,j]));assert len(row)==1;native.append(int(row[0]))
        sequence=''.join(AA[i] for i in native)
        if name=='pdz2':assert sequence=='HEDAVAALK'
        if name=='pdz3':assert sequence=='HEQAAIALK'
        theta,single_error,energy=fit_calibration(o);wt=o.selected_wt_counts/o.unselected_wt_counts
        pairs=[]
        for i in range(9):
            for j in range(i+1,9):
                for a in range(20):
                    for b in range(20):
                        if a==native[i] or b==native[j]:continue
                        y=o.deltaG_double[i,j,a,b];n=o.unselected_double_counts[i,j,a,b];s=o.selected_double_counts[i,j,a,b]
                        if np.isfinite(y) and np.isfinite(n) and n>0 and np.isfinite(s):pairs.append((i,a,j,b,float(y),float(n),float(s)))
        frame=pd.DataFrame(pairs,columns=['i','a','j','b','G','input','selected'])
        x=np.log(frame.selected.to_numpy()/frame.input.to_numpy()/wt);mask=np.isfinite(x)
        double_error=float(np.sqrt(np.mean((energy(x[mask],theta)-frame.G.to_numpy()[mask])**2)))
        confidence_available=single_error<=.001 and double_error<=.001
        calibration.append(dict(domain=name,alpha=float(theta[0]),fraction_WT=float(theta[1]),single_RMSE_kcal=single_error,finite_double_RMSE_kcal=double_error,count_support_available=confidence_available,WT_Kd_M=float(o.Kd),native_helix=sequence))
        frame['affinity']=np.exp(-frame.G/RT)
        if confidence_available:
            raw_low=low_ratio(frame.selected.to_numpy(),frame.input.to_numpy(),wt)
            threshold=(.5/(1-theta[1]+.5*theta[1]))**(1/theta[0]);frame['count_supported']=raw_low>=threshold
        else:frame['count_supported']=False
        single=[]
        for j in range(9):
            for a in range(20):
                if a==native[j] or not np.isfinite(o.deltaG_single[a,j]):continue
                single.append((j,a,float(o.deltaG_single[a,j]),float(o.unselected_single_counts[a,j]),float(o.selected_single_counts[a,j])))
        s=pd.DataFrame(single,columns=['site','aa','G','input','selected']);s['affinity']=np.exp(-s.G/RT)
        if confidence_available:s['harmful_confident']=high_ratio(s.selected.to_numpy(),s.input.to_numpy(),wt)<1
        else:s['harmful_confident']=False
        mapping={(int(r.site),int(r.aa)):r for r in s.itertuples()}
        damages=s[s.affinity<=.2];cover.append(dict(domain=name,split='development',helix=sequence,finite_single_energies=len(s),designed_nonWT_single=171,finite_double_energies=len(frame),designed_double=12996,damage_tasks=len(damages),count_support_available=confidence_available))
        for d in damages.itertuples():
            hits=frame[((frame.i==d.site)&(frame.a==d.aa))|((frame.j==d.site)&(frame.b==d.aa))].copy()
            hits['rsite']=np.where((hits.i==d.site)&(hits.a==d.aa),hits.j,hits.i);hits['raa']=np.where((hits.i==d.site)&(hits.a==d.aa),hits.b,hits.a)
            hits=hits[[key in mapping for key in zip(hits.rsite,hits.raa)]].copy()
            hits['R_affinity']=[mapping[key].affinity for key in zip(hits.rsite,hits.raa)]
            hits['R_confident_harmful']=[mapping[key].harmful_confident for key in zip(hits.rsite,hits.raa)]
            for regime in ['all_R','point_harmful_R','confident_harmful_R']:
                f=hits.copy()
                if regime=='point_harmful_R':f=f[f.R_affinity<1]
                if regime=='confident_harmful_R':f=f[f.R_confident_harmful]
                f=f.sort_values(['R_affinity','rsite','raa'],ascending=[False,True,True])
                record=dict(domain=name,damage=f'{sequence[d.site]}{d.site+1}{AA[d.aa]}',damage_site=d.site+1,regime=regime,candidates=len(f),point_reachable=bool((f.affinity>=.5).any()),count_supported_reachable=bool(f.count_supported.any()),uncertainty_available=confidence_available)
                for k in [8,32]:record[f'WT_point{k}']=bool((f.head(k).affinity>=.5).any());record[f'WT_count_supported{k}']=bool(f.head(k).count_supported.any())
                records.append(record)
                for row in f[(f.affinity>=.5)&(f.R_affinity<1)].head(1).itertuples():examples.append(dict(domain=name,damage=record['damage'],rescue=f'{sequence[row.rsite]}{row.rsite+1}{AA[row.raa]}',D_affinity=d.affinity,R_affinity=row.R_affinity,combo_affinity=row.affinity,count_supported=row.count_supported,regime=regime))
    tasks=pd.DataFrame(records);tasks.to_csv(OUT/'pdz2018_tasks.csv',index=False);pd.DataFrame(cover).to_csv(OUT/'pdz2018_coverage.csv',index=False);pd.DataFrame(calibration).to_csv(OUT/'pdz2018_calibration_QC.csv',index=False);pd.DataFrame(examples).to_csv(OUT/'pdz2018_development_examples.csv',index=False)
    summary=tasks.groupby(['domain','regime']).agg(tasks=('damage','size'),candidate_median=('candidates','median'),tasks_gt32=('candidates',lambda x:int((x>32).sum())),point_reachable=('point_reachable','sum'),count_supported_reachable=('count_supported_reachable','sum'),WT_point32=('WT_point32','sum'),WT_count_supported32=('WT_count_supported32','sum')).reset_index()
    summary.to_csv(OUT/'pdz2018_summary.csv',index=False)
    audit=dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),loaded_domains=names,unloaded_test_domains=['shank3','zo1'],test_double_outcomes_analyzed=0,energy_NaN='unresolved/noninvertible energy excluded and reported; raw-count presence does not create a finite affinity label',secondary='conditional count support only for single/double calibration consistency; fixed WT, no biological variance or calibration uncertainty',new_model_runs=0)
    (OUT/'pdz2018_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(pd.DataFrame(calibration).to_string(index=False));print(summary.to_string(index=False))


if __name__=='__main__':main()
