"""Verify every audit catalogue row, pairing count and split exclusion."""
import json
import numpy as np
import pandas as pd
from pdz2018_data import ROOT,dump
from audit_pdz_auxiliary import OUT,BASE,LIGANDS,ALIGN,GENES,files,check,fasta


def main():
    check();assert json.loads((OUT/'execution.json').read_text())['status']=='completed'
    cat=pd.read_csv(OUT/'identity_catalog.csv.gz',keep_default_na=False);stats=pd.read_csv(OUT/'file_audit.csv');pair=pd.read_csv(OUT/'paired_phenotypes.csv')
    assert set(cat.source)=={str(f['path'].relative_to(ROOT)) for f in files()}
    assert not cat.source.str.contains('erbin|shank|zo1',case=False).any()
    assert not cat.duplicated(['source','mutant']).any()
    sets={};nt={};checked=0
    for info in files():
        source=str(info['path'].relative_to(ROOT));subset=cat[cat.source==source];f=pd.read_csv(info['path']);ref=f.loc[f.Nham_aa==0,'aa_seq'].unique();assert len(ref)==1;ref=ref[0]
        assert ref==fasta()[ALIGN[info['domain']]]
        eligible=f[(~f.indel)&(~f.STOP)&(~f.STOP_readthrough)&f.Nham_aa.isin([1,2])].copy()
        def mutate(s):
            if len(s)!=len(ref):return None
            aa=set('ACDEFGHIKLMNPQRSTVWY');changed=[(i+1,a,b) for i,(a,b) in enumerate(zip(ref,s)) if a!=b]
            if any(a not in aa or b not in aa for i,a,b in changed):return None
            return ':'.join(a+str(i)+b for i,a,b in changed)
        eligible['key']=eligible.aa_seq.map(mutate);eligible=eligible[eligible.key.notna()].copy()
        eligible['input_sum']=eligible[[f'count_e{i}_s0' for i in [1,2,3]]].sum(axis=1)
        maxima=eligible.groupby('key').input_sum.max();assert set(maxima.index)==set(subset.mutant)
        preferred=eligible.sort_values(['key','input_sum','nt_seq'],ascending=[True,False,True],kind='stable').drop_duplicates('key').reset_index().set_index('key')['index'].to_dict()
        for row in subset.itertuples():
            raw=f.loc[row.row_index];assert mutate(raw.aa_seq)==row.mutant
            assert np.isclose(row.input_count,maxima[row.mutant])
            assert row.row_index==preferred[row.mutant]
            assert bool(row.finite_aggregate)==bool(np.isfinite(raw.fitness) and np.isfinite(raw.sigma) and raw.sigma>=0)
            assert bool(row.finite_replicate3)==bool(np.isfinite([raw[f'fitness{i}_uncorr'] for i in [1,2,3]]).all())
            checked+=1
        key=(info['assay'],info['block']);sets[key]=set(subset.loc[subset.finite_aggregate,'mutant'])
        good=eligible[np.isfinite(eligible.fitness)&np.isfinite(eligible.sigma)&(eligible.sigma>=0)]
        nt[key]=set(zip(good.key,good.nt_seq))
        stat=stats[stats.source==source].iloc[0];assert stat.unique_AA==len(subset) and stat.finite_aggregate==subset.finite_aggregate.sum()
        print('Verified source',info['path'].name,flush=True)
    for row in pair.itertuples():
        a=row.domain;b=row.domain+'_'+row.ligand;common=set();nts=set();n=0
        for block in ['b1','b2']:
            if (a,block) not in sets:continue
            common|=sets[a,block]&sets[b,block];joint=nt[a,block]&nt[b,block];nts|={k for k,_ in joint};n+=len(joint)
        assert len(common)==row.matched_AA and len(nts)==row.matched_AA_with_common_nt and n==row.common_block_nt_records
    maps=json.loads((OUT/'sequence_mapping.json').read_text())
    for row in maps:
        seq=fasta()[ALIGN[row['domain']]].rstrip('*');assert seq==row['actual_sequence']
        assert ''.join(seq[i-1] for i in row['aux_positions'])==row['helix']
        assert row['shared_alignment_columns']==list(range(82,91))
    eligible=cat[cat.finite_aggregate];splits=pd.read_csv(OUT/'split_eligibility.csv')
    for row in splits.itertuples():
        if row.regime=='strict_aligned_site':mask=~eligible.scan_sites.map(lambda s:str(row.held) in s.split(';'))
        elif row.regime=='leave_domain':mask=eligible.domain!=row.held
        else:mask=eligible.gene!=row.held
        train=eligible[mask];assert len(train)==row.training_rows and (~mask).sum()==row.excluded_rows
        assert len(train[['domain','mutant']].drop_duplicates())==row.unique_domain_variant_pairs
    check();dump(OUT/'verification.json',dict(status='passed',allowlisted_development_files=18,catalog_rows_independently_checked=checked,
        representative_choice_outcome_independent=True,paired_phenotype_links_verified=len(pair),aligned_site_maps_verified=4,
        split_exclusions_verified=len(splits),heldout_outcome_files_read=0,new_model_fits=0,old_artifacts_unchanged=True))
    print('Auxiliary audit verified',checked,'identity rows',flush=True)


if __name__=='__main__':main()
