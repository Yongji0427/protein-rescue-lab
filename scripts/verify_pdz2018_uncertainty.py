"""Independent interval, source-count and unchanged-trajectory checks."""
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest,beta
from scipy.io import loadmat
from pdz2018_data import ROOT,AA,DOMAINS,RT,dump
from pdz2018_uncertainty import count_ratio_interval,relative_interval,score_counts,TAIL_ALPHA,DESIGNED_FAMILY,missing_pdz3

OUT=ROOT/'experiments/pdz2018_uncertainty_v1'
PILOT=ROOT/'experiments/pdz2018_point_dev_v1'


def main():
    for base in [OUT/'freeze.json',ROOT/'experiments/gb1_final_holdout_v1/freeze.json']:
        manifest=json.loads(base.read_text())['hashes']
        assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h for path,h in manifest.items())
    old=json.loads((PILOT/'final_artifact_hashes.json').read_text())
    assert all(hashlib.sha256((PILOT/path).read_bytes()).hexdigest()==h for path,h in old.items())
    effective={**json.loads((PILOT/'freeze.json').read_text())['hashes'],**json.loads((PILOT/'setup_addendum.json').read_text())['hashes']}
    assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h for path,h in effective.items())
    comparisons=0
    for s,n in [(0,20),(1,50),(12,100),(100,100),(1000,100),(81763,58124)]:
        q=binomtest(s,s+n).proportion_ci(confidence_level=1-2*TAIL_ALPHA,method='exact')
        expected_low=q.low/(1-q.low);expected_high=q.high/(1-q.high)
        low,high=count_ratio_interval(s,n)
        assert np.isclose(low,expected_low,rtol=1e-6,atol=1e-9)
        assert np.isclose(high,expected_high,rtol=1e-6,atol=1e-9)
        comparisons+=1
    for s,n in [(-1,20),(1.5,20),(np.nan,20)]:
        try:count_ratio_interval(s,n);raise AssertionError('Invalid counts accepted')
        except ValueError:pass
    low,high=count_ratio_interval(0,0);assert low==0 and np.isinf(high)
    unknown=missing_pdz3();assert unknown['status']=='unavailable' and unknown['count_lower_affinity'] is None and not unknown['biological_interval_available']
    raw_lo,raw_hi=count_ratio_interval(100,500);lo,hi=relative_interval(100,500,1000,2000)
    assert lo<raw_lo/(1000/2000) and hi>raw_hi/(1000/2000)
    assert DESIGNED_FAMILY==39504 and np.isclose(2*DESIGNED_FAMILY*TAIL_ALPHA,.05)
    # Rebuild every available transformed interval directly from source counts.
    source=loadmat(ROOT/'data/pdz2018/source/Salinas_Ranganathan_data.mat',variable_names=list(DOMAINS),squeeze_me=True,struct_as_record=False)
    assert 'shank3' not in source and 'zo1' not in source
    cal=pd.read_csv(ROOT/'experiments/dataset_feasibility_v1/pdz2018_calibration_QC.csv').set_index('domain')
    rows=pd.read_csv(OUT/'double_bounds.csv');checked=0
    for domain,f in rows.groupby('domain'):
        if domain=='pdz3':
            assert f.counting_lower.isna().all() and f.combo_supported.isna().all();continue
        o=source[domain];alpha,fw=cal.loc[domain,['alpha','fraction_WT']]
        indices=[]
        for d,r in zip(f.mutation1,f.mutation2):indices.append((int(d[1:-1])-1,int(r[1:-1])-1,AA.index(d[-1]),AA.index(r[-1])))
        n=np.array([o.unselected_double_counts[k] for k in indices]);s=np.array([o.selected_double_counts[k] for k in indices])
        # Direct Clopper-Pearson beta quantiles with WT sampled separately.
        wl=beta.ppf(TAIL_ALPHA,o.selected_wt_counts,o.unselected_wt_counts+1)
        wu=beta.ppf(1-TAIL_ALPHA,o.selected_wt_counts+1,o.unselected_wt_counts)
        ql=np.zeros(len(s));positive=s>0;ql[positive]=beta.ppf(TAIL_ALPHA,s[positive],n[positive]+1)
        qu=beta.ppf(1-TAIL_ALPHA,s+1,n)
        rl=(ql/(1-ql))/(wu/(1-wu));rh=(qu/(1-qu))/(wl/(1-wl))
        def energy_affinity(x):
            fm=fw*x**alpha
            g=np.full(len(fm),-np.inf);valid=fm<1
            with np.errstate(divide='ignore'):
                g[valid]=RT*(np.log((1-fm[valid])/fm[valid])-np.log((1-fw)/fw))
            return np.exp(-g/RT)
        assert np.allclose(energy_affinity(rl),f.counting_lower,rtol=1e-11,atol=1e-12)
        assert np.allclose(energy_affinity(rh),f.counting_upper,rtol=1e-11,atol=1e-12)
        expected=(f.point_affinity>=.5)&(f.counting_lower>=.5)
        assert np.array_equal(expected,f.combo_supported.astype(bool));checked+=len(f)
    # Re-score every model and deterministic WT/fixed trajectory independently.
    bounds={};singles=pd.read_csv(OUT/'single_bounds.csv');smap={(r.domain,r.mutation):r for r in singles.itertuples()}
    for r in rows.itertuples():
        bounds[r.domain,r.mutation1,r.mutation2]=r;bounds[r.domain,r.mutation2,r.mutation1]=r
    core=pd.read_csv(OUT/'core_rescored.csv')
    fixed={}
    for chunk in pd.read_csv(PILOT/'fixed_query_log.csv.gz',chunksize=100000):
        for (task_id,method),f in chunk[chunk.repeat==0].groupby(['task_id','method']):fixed.setdefault((task_id,method),[]).extend(f.rescue.tolist())
    for row in core.itertuples():
        domain,d=row.task_id.split('__')
        if domain=='pdz3':assert pd.isna(row.combo_supported) and not row.count_available;continue
        if row.method=='Agnes_AGH':ids=[r['rescue'] for r in json.loads((PILOT/'queried_results'/(row.task_id+'.json')).read_text())['queries']]
        else:ids=fixed[row.task_id,row.method]
        success=any(bool(bounds[domain,d,r].combo_supported) for r in ids)
        assert success==row.combo_supported
        assert bool(success and smap[domain,d].damage_confirmed)==row.full_task_supported
        assert any(bool(bounds[domain,d,r].combo_supported) and bool(smap[domain,r].harmful_confirmed) for r in ids)==row.harmful_R_combo_supported
    dump(OUT/'verification.json',dict(status='passed',independent_exact_binomial_comparisons=comparisons,
        available_double_bounds_rebuilt=checked,core_trajectories_checked=len(core),
        old_GB1_execution_hashes_unchanged=30,old_PDZ_results_and_execution_unchanged=True,
        same_query_trajectories=True,new_model_calls=0,new_queries=0,test_arrays_loaded=False,
        missing_PDZ3_preserved=True,scope='conditional simultaneous counting only; biological and calibration uncertainty unresolved'))
    print('Verified:',checked,'double intervals and',len(core),'matched core trajectories')


if __name__=='__main__':main()
