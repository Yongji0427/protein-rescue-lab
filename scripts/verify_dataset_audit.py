"""Independent reconstruction and provenance/isolation checks for feasibility audit."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/dataset_feasibility_v1'


def main():
    freeze=json.loads((ROOT/'experiments/gb1_final_holdout_v1/freeze.json').read_text())
    assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest for path,digest in freeze['hashes'].items())
    for name,digest in json.loads((OUT/'pre_double_analysis_freeze.json').read_text())['hashes'].items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest
    mat=ROOT/'data/pdz2018/source/Salinas_Ranganathan_data.mat';assert hashlib.md5(mat.read_bytes()).hexdigest()=='3c477dfd0d5b4e24891feae3dbaa525e'
    audit=json.loads((OUT/'pdz2018_audit.json').read_text());assert audit['loaded_domains']==['pdz2','pdz3','syntrophin'];assert audit['test_double_outcomes_analyzed']==0
    data=loadmat(mat,variable_names=audit['loaded_domains'],squeeze_me=True,struct_as_record=False)
    identities=0
    for name in audit['loaded_domains']:
        o=data[name]
        expected=o.deltaG_single.T[:,None,:,None]+o.deltaG_single.T[None,:,None,:]-o.deltaG_double
        mask=np.isfinite(expected)&np.isfinite(o.deltadeltaG)
        assert np.allclose(expected[mask],o.deltadeltaG[mask],atol=1e-8),name
        identities+=int(mask.sum())
    p26=json.loads((OUT/'pdz2026_audit.json').read_text());assert p26['test_double_outcomes_analyzed']==0 and all('erbin' not in x['source'] for x in p26['development_sources'])
    pub=ROOT/'data/pdz2026/public'
    f=pd.read_csv(pub/'erbin_pdz1_cript_double_identity.csv');assert not set(f.columns)&{'fitness','sigma','activity','lower'}
    for source in (ROOT/'data/pdz2026/source/extracted').rglob('*.provenance.json'):
        proof=json.loads(source.read_text());file=Path(str(source).replace('.provenance.json',''));assert hashlib.sha256(file.read_bytes()).hexdigest()==proof['sha256']
    # WT rankings are recomputed from public single values, not copied from outcome summaries.
    t=pd.read_csv(OUT/'pdz2026_tasks.csv');examples=0
    cal=pd.read_csv(OUT/'pdz2026_public_calibration.csv').set_index('source')
    for assay in t.assay.unique():
        s=pd.read_csv(pub/(assay+'_singles.csv')).set_index('mutant');d=pd.read_csv(pub/(assay+'_double_identity.csv'))
        d['y']=np.nan;d['lo']=np.nan
        for source,index in d.groupby('source').groups.items():
            raw=pd.read_csv(ROOT/source,usecols=['fitness','sigma']);c=cal.loc[source];ri=d.loc[index,'row_index'].to_numpy(int)
            d.loc[index,'y']=(raw.loc[ri,'fitness'].to_numpy()-c.stop_single_median)/c.dynamic_range
            d.loc[index,'lo']=d.loc[index,'y']-1.6448536269514722*raw.loc[ri,'sigma'].to_numpy()/c.dynamic_range
        s=s[np.isfinite(s.activity)&np.isfinite(s.activity_sigma)&(s.activity_sigma>=0)]
        for row in t[(t.assay==assay)&(t.regime=='confident_harmful_R')].itertuples():
            pairs=d[d.mutant.str.split(':').map(lambda pair:row.damage in pair)].copy()
            pairs['R']=pairs.mutant.str.split(':').map(lambda pair:next(r for r in pair if r!=row.damage))
            pairs=pairs[pairs.R.isin(s.index)];pairs=pairs[pairs.R.map(s.activity+1.6448536269514722*s.activity_sigma)<1]
            pairs=pairs[np.isfinite(pairs.y)&np.isfinite(pairs.lo)];pairs['WT']=pairs.R.map(s.activity)
            pairs=pairs.sort_values(['WT','R'],ascending=[False,True])
            assert len(pairs)==row.candidates and bool((pairs.lo>=.5).any())==row.interval_supported_reachable
            assert bool((pairs.head(32).lo>=.5).any())==row.WT_supported32
            examples+=1
    result=dict(status='passed',checks=['all 30 old GB1 execution hashes unchanged','new pre-double-analysis rules unchanged','2018 publisher MAT equals Dryad checksum; only three development domains loaded','published single/double/coupling energies independently satisfy cycle identity','all selectively fetched Zenodo files CRC verified at extraction and SHA rechecked','held ERBIN identity catalog has no outcome columns; held double rows absent from analysis','every confident-harmful-R 2026 task rebuilt and WT32/support compared independently'],energy_identities_checked=identities,PDZ2026_tasks_independently_rebuilt=examples)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
