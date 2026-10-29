"""Prepare public controls/singles and identity-only graphs before double outcomes."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/dataset_feasibility_v1';BASE=ROOT/'data/pdz2026/source/extracted/data/inputs_mochi'
ASSAYS=['psd95_pdz3_cript','psd95_pdz2_cript','psd95_pdz2_nmdar2a','nherf3_pdz1_uhmk1','nherf3_pdz1_dap-1','nherf3_pdz2_cep164','erbin_pdz1_cript']


def mutation_ids(seq,ref):
    assert len(seq)==len(ref)
    changes=[f'{a}{i+1}{b}' for i,(a,b) in enumerate(zip(ref,seq)) if a!=b]
    if any(v[0] not in 'ACDEFGHIKLMNPQRSTVWY' or v[-1] not in 'ACDEFGHIKLMNPQRSTVWY' for v in changes):return None
    return ':'.join(changes)


def main():
    public=ROOT/'data/pdz2026/public';public.mkdir(exist_ok=True)
    calibration=[];cover=[];inputs=[]
    for assay in ASSAYS:
        identity=[];singles=[];reference=None
        files=sorted(BASE.glob(assay+'_b*_all_variants.txt'))
        assert files
        for path in files:
            # Only identity, pre-selection counts and public controls/singles are inspected.
            meta=pd.read_csv(path,usecols=['aa_seq','Nham_aa','indel','STOP','WT','count_e1_s0','count_e2_s0','count_e3_s0'])
            values=pd.read_csv(path,usecols=['fitness','sigma'])
            controls=meta.Nham_aa<=1
            public_values=values.loc[controls].copy()
            refs=meta.loc[meta.Nham_aa==0,'aa_seq'].unique();assert len(refs)==1
            ref=refs[0]
            if reference is not None:assert reference==ref
            reference=ref
            wt=float(public_values.loc[meta.loc[controls,'Nham_aa']==0,'fitness'].median())
            stop=float(public_values.loc[meta.loc[controls,'STOP'] & (meta.loc[controls,'Nham_aa']==1),'fitness'].median())
            assert np.isfinite(wt) and np.isfinite(stop) and wt>stop
            block=path.name.split('_')[-3]
            calibration.append(dict(assay=assay,block=block,WT_synonymous_median=wt,stop_single_median=stop,dynamic_range=wt-stop,WT_controls=int((meta.Nham_aa==0).sum()),single_stop_controls=int((meta.STOP&(meta.Nham_aa==1)).sum()),source=str(path.relative_to(ROOT))))
            ids=meta.aa_seq.map(lambda seq:mutation_ids(seq,ref))
            catalog=pd.DataFrame({'mutant':ids,'input_count':meta[['count_e1_s0','count_e2_s0','count_e3_s0']].sum(axis=1),'row_index':meta.index,'source':str(path.relative_to(ROOT)),'block':block})
            catalog=catalog[(~meta.indel)&(~meta.STOP)&ids.notna()&meta.Nham_aa.isin([1,2])].copy()
            n=ids.str.count(':')+1;assert (n.loc[catalog.index].to_numpy()==meta.loc[catalog.index,'Nham_aa'].to_numpy()).all()
            s=catalog[meta.loc[catalog.index,'Nham_aa']==1].copy();s['fitness']=public_values.loc[s.index,'fitness'];s['sigma']=public_values.loc[s.index,'sigma']
            s['activity']=(s.fitness-stop)/(wt-stop);s['activity_sigma']=s.sigma/(wt-stop)
            singles.append(s);identity.append(catalog[meta.loc[catalog.index,'Nham_aa']==2])
            inputs.append({'source':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'assay':assay,'block':block})
        single=pd.concat(singles).sort_values(['input_count','source','row_index'],ascending=[False,True,True]).drop_duplicates('mutant').sort_values('mutant')
        double=pd.concat(identity).sort_values(['input_count','source','row_index'],ascending=[False,True,True]).drop_duplicates('mutant').sort_values('mutant')
        single.to_csv(public/(assay+'_singles.csv'),index=False);double.to_csv(public/(assay+'_double_identity.csv'),index=False)
        parts=double.mutant.str.split(':',expand=True)
        counts=pd.concat([parts[0],parts[1]]).value_counts()
        damaged=single[np.isfinite(single.activity)&(single.activity<=.2)]
        candidate=damaged.mutant.map(counts).fillna(0)
        positions=pd.concat([parts[0],parts[1]]).str[1:-1].astype(int).value_counts()
        cover.append(dict(assay=assay,split='test_unread' if assay.startswith('erbin') else 'development',WT_sequence=reference,singles=len(single),unique_double_identities=len(double),damage_singles=len(damaged),candidate_min=int(candidate.min()) if len(candidate) else 0,candidate_median=float(candidate.median()) if len(candidate) else 0,candidate_max=int(candidate.max()) if len(candidate) else 0,tasks_gt32=int((candidate>32).sum()),most_connected_positions=positions.head(3).to_dict()))
    pd.DataFrame(calibration).to_csv(OUT/'pdz2026_public_calibration.csv',index=False);pd.DataFrame(cover).to_json(OUT/'pdz2026_identity_coverage.json',orient='records',indent=2)
    rules=dict(representative='one nucleotide variant/block per AA genotype, selected only by maximum summed pre-selection input count; tie source then row index; no outcome-based pooling',normalization='per-block public synonymous-WT median minus single-stop median; fixed controls for intervals; selection proxy, not affinity',uncertainty='normal 90% interval using supplied DiMSum sigma, with fixed controls; excludes normalization uncertainty; no independent Poisson assumption',temperature_for_2018='25 Celsius=298.15K from experimental Methods',preserved_test='ERBIN double outcomes not selected or inspected',inputs=inputs)
    (OUT/'calibration_and_representation_rules.json').write_text(json.dumps(rules,indent=2)+'\n')
    print(pd.DataFrame(cover).drop(columns=['WT_sequence']).to_string(index=False))


if __name__=='__main__':main()
