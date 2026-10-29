"""Audit paired functional data and fold-safe identity mappings; no training."""
from datetime import datetime,timezone
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from pdz2018_data import ROOT,DOMAINS,hashes,dump

OUT=ROOT/'experiments/pdz_auxiliary_audit_v1'
BASE=ROOT/'data/pdz2026/source/extracted/data'
LIGANDS={'psd95_pdz2':['cript','nmdar2a'],'psd95_pdz3':['cript'],'nherf3_pdz1':['uhmk1','dap-1'],'nherf3_pdz2':['cep164']}
ALIGN={'psd95_pdz2':'DLG4_PDZ2','psd95_pdz3':'DLG4_PDZ3','nherf3_pdz1':'NHRF3_PDZ1','nherf3_pdz2':'NHRF3_PDZ2'}
GENES={d:'DLG4' if d.startswith('psd95') else 'PDZK1' for d in LIGANDS}
AA=set('ACDEFGHIKLMNPQRSTVWY')


def files():
    result=[]
    for domain,ligands in LIGANDS.items():
        for ligand in [None]+ligands:
            assay=domain+('_'+ligand if ligand else '')
            for p in sorted((BASE/'inputs_mochi').glob(assay+'_b*_all_variants.txt')):
                assert not any(x in p.name.lower() for x in ['erbin','shank','zo1'])
                result.append(dict(path=p,domain=domain,gene=GENES[domain],ligand=ligand or '',phenotype='binding' if ligand else 'abundance',assay=assay,block=p.name.split('_')[-3]))
    assert len(result)==18
    return result


def freeze():
    assert not (OUT/'freeze.json').exists()
    p=dict(version=OUT.name,scope='Auxiliary data eligibility and interface audit; no fitted model, new benchmark or predictor claims',
        allowlist=[str(f['path'].relative_to(ROOT)) for f in files()],excluded_outcomes=['ERBIN','SHANK3','ZO1'],
        phenotypes='aPCA abundance andbPCA binding; DiMSum fitness andsigma remain assay-specific; not affinity fractions or directly interchangeable kcal/mol',
        valid_identity='No indels or STOP/readthrough, Nham_aa in1,2; native residues checked against actual file WT, standard20AA substitutions only',
        dedup='Within each file and AA genotype choose highest summed preselection count, tie nt_seq then originalrow; no outcome-based representative choice; synonymous encodings and blocks are not biological replicate units',
        paired='Report matched AA identities and exact(block,nt_seq) overlaps; common phenotype availability and assay scales separately',
        QC='Report finite aggregate fitness/sigma, finite3replicate fitness, raw counts and descriptive replicate correlations; do not select by rescue outcomes',
        sequence='Actual WT in processed tables must match author FASTA; discrepancies against XLSX/structural alignment logged; scan9 mapping via public author structural alignment verified by nativeAA',
        leakage='For strict auxiliary supervised training exclude every mutation touching held aligned site, for bothsingle/double and all phenotypes/ligands; current public2018 singles remain permitted. External single readouts require separately declared equal-access reference regime.',
        transfer='Leave-domain-out removes all auxiliary labels for that domain; parent-protein generalization must group DLG4 PDZ2+PDZ3 and PDZK1 PDZ1+PDZ2 across studies/species',
        reuse='2026 PSD95-PDZ3 reuses Faure2022 measurements; not an independent validation source. Do not load or use author globally-fitted MoCHI weights/energies.',
        selection='No outcome-selected cohorts; alldevelopment sources audited; 2026 sparse candidate-pool verdict retained',
        next='Deliver eligibility, pairing, mapping, split counts and unresolved-items report for discussion; no training')
    dump(OUT/'protocol.json',p)
    preserved={}
    for folder in ['dataset_feasibility_v1','pdz2018_background_v1','pdz2018_rescue_ranking_v1','pdz2018_esm_v2','pdz2018_portfolio_v1','pdz2018_two_checks_v1']:
        preserved.update(hashes([x for x in (ROOT/'experiments'/folder).rglob('*') if x.is_file()]))
    dump(OUT/'preserved_hashes.json',preserved)
    inputs=[f['path'] for f in files()]
    inputs += [BASE/'assays_info.xlsx',BASE/'annotations/structural_alignment_tcoffee.csv',BASE/'fasta_sequences/fasta_sequences.fasta',
               OUT/'protocol.json',OUT/'preserved_hashes.json']
    inputs += [ROOT/'scripts'/n for n in ['audit_pdz_auxiliary.py','verify_pdz_auxiliary.py','report_pdz_auxiliary.py']]
    inputs += [p for p in (OUT/'sources').glob('*') if p.is_file()]
    dump(OUT/'freeze.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes(inputs)))


def check():
    for name in ['freeze.json','preserved_hashes.json']:
        o=json.loads((OUT/name).read_text())
        for p,h in o.get('hashes',o).items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p


def fasta():
    seq={};key=None
    for line in (BASE/'fasta_sequences/fasta_sequences.fasta').read_text().splitlines():
        if line.startswith('>'):key=line[1:].upper().replace('-','_');seq[key]=''
        elif key:seq[key]+=line.strip().upper()
    return seq


def read_source(info):
    p=info['path'];assert p in [f['path'] for f in files()]
    provenance=json.loads(p.with_suffix(p.suffix+'.provenance.json').read_text())
    assert hashlib.sha256(p.read_bytes()).hexdigest()==provenance['sha256']
    columns=['nt_seq','aa_seq','Nham_aa','WT','indel','STOP','STOP_readthrough','fitness','sigma']
    columns += [f'count_e{i}_s{s}' for i in [1,2,3] for s in [0,1]]
    columns += [f'fitness{i}_uncorr' for i in [1,2,3]]+[f'sigma{i}_uncorr' for i in [1,2,3]]
    f=pd.read_csv(p,usecols=columns);ref=f.loc[f.Nham_aa==0,'aa_seq'].unique();assert len(ref)==1;ref=str(ref[0])
    assert ref==fasta()[ALIGN[info['domain']]]
    def variant(seq):
        if len(seq)!=len(ref):return None
        changes=[f'{a}{i+1}{b}' for i,(a,b) in enumerate(zip(ref,seq)) if a!=b]
        if not all(c[0] in AA and c[-1] in AA for c in changes):return None
        return ':'.join(changes)
    valid=(~f.indel)&(~f.STOP)&(~f.STOP_readthrough)&f.Nham_aa.isin([1,2]);g=f.loc[valid].copy()
    g['mutant']=g.aa_seq.map(variant);g=g[g.mutant.notna()].copy();assert np.array_equal(g.mutant.str.count(':')+1,g.Nham_aa)
    g['row_index']=g.index;g['input_count']=g[[f'count_e{i}_s0' for i in [1,2,3]]].sum(axis=1)
    g['finite_aggregate']=np.isfinite(g.fitness)&np.isfinite(g.sigma)&(g.sigma>=0)
    g['finite_replicate3']=np.isfinite(g[[f'fitness{i}_uncorr' for i in [1,2,3]]]).all(axis=1)
    reps=g.sort_values(['input_count','nt_seq','row_index'],ascending=[False,True,True]).drop_duplicates('mutant').copy()
    return ref,g,reps,len(f)


def run():
    check();assert not (OUT/'started.json').exists();dump(OUT/'started.json',dict(at_utc=datetime.now(timezone.utc).isoformat()))
    metadata=pd.read_excel(BASE/'assays_info.xlsx');metadata=metadata[metadata.pdz_name.str.lower().isin(LIGANDS)].copy()
    alignment=pd.read_csv(BASE/'annotations/structural_alignment_tcoffee.csv',sep=r'\s+',usecols=['pdz_name','WT','structural_alignment_pos','Pos'])
    alignment=alignment[alignment.pdz_name.isin(ALIGN.values())]
    file_stats=[];catalog=[];maps=[];variants={};ntsets={};finite_sets={};refs={};helixmaps={}
    # The shared9 columns are established by exact2018 sequences in bothDLG4 domains.
    aligned=[]
    for domain,old in [('psd95_pdz2','pdz2'),('psd95_pdz3','pdz3')]:
        ref=fasta()[ALIGN[domain]].rstrip('*');helix=DOMAINS[old][0];assert ref.count(helix)==1
        positions=list(range(ref.index(helix)+1,ref.index(helix)+10));a=alignment[alignment.pdz_name==ALIGN[domain]].set_index('Pos')
        aligned.append([int(a.loc[p,'structural_alignment_pos']) for p in positions])
    assert aligned[0]==aligned[1]
    for domain in LIGANDS:
        ref=fasta()[ALIGN[domain]].rstrip('*');a=alignment[alignment.pdz_name==ALIGN[domain]]
        h=a[a.structural_alignment_pos.isin(aligned[0])].sort_values('structural_alignment_pos');assert len(h)==9 and h.Pos.notna().all()
        assert all(ref[int(r.Pos)-1]==r.WT for r in h.itertuples())
        helixmaps[domain]={int(r.Pos):i+1 for i,r in enumerate(h.itertuples())}
        meta=metadata[metadata.pdz_name.str.lower()==domain].iloc[0]
        alignment_seq=''.join(a.loc[a.Pos.notna(),'WT']);mismatch=[i+1 for i,(x,y) in enumerate(zip(ref,alignment_seq)) if x!=y]
        assert not mismatch
        maps.append(dict(domain=domain,parent_gene=GENES[domain],actual_sequence=ref,actual_length=len(ref),metadata_sequence=meta.pdz_sequence,
            metadata_matches=ref==meta.pdz_sequence,author_fasta_matches=True,structural_alignment_prefix_matches=True,
            unmapped_alignment_tail=alignment_seq[len(ref):],aux_positions=list(helixmaps[domain]),old_scan_positions=list(range(1,10)),
            shared_alignment_columns=aligned[0],helix=''.join(ref[p-1] for p in helixmaps[domain]),
            full2018_assay_construct_match='unconfirmed; only scan9 native identities verified for DLG4 domains'))
    dump(OUT/'sequence_mapping.json',maps)
    for info in files():
        ref,g,r,total=read_source(info);domain=info['domain'];key=(info['assay'],info['block']);refs[domain]=ref
        positions=r.mutant.map(lambda s:[int(m[1:-1]) for m in s.split(':')])
        finite=r[r.finite_aggregate];variants[key]=set(r.mutant);finite_sets[key]=set(finite.mutant)
        ntsets[key]=set(zip(g.loc[g.finite_aggregate,'mutant'],g.loc[g.finite_aggregate,'nt_seq']))
        corrs={}
        for i,j in [(1,2),(1,3),(2,3)]:
            a=r[[f'fitness{i}_uncorr',f'fitness{j}_uncorr']].replace([np.inf,-np.inf],np.nan).dropna()
            corrs[f'replicate_r{i}{j}']=float(a.corr().iloc[0,1]);corrs[f'replicate_n{i}{j}']=len(a)
        countcols=[f'count_e{i}_s{s}' for i in [1,2,3] for s in [0,1]]
        file_stats.append(dict(source=str(info['path'].relative_to(ROOT)),domain=domain,gene=info['gene'],assay=info['assay'],phenotype=info['phenotype'],ligand=info['ligand'],block=info['block'],
            rows=total,valid_nt_rows=len(g),unique_AA=len(r),single_AA=int((r.Nham_aa==1).sum()),double_AA=int((r.Nham_aa==2).sum()),
            finite_aggregate=int(r.finite_aggregate.sum()),finite_replicate3=int(r.finite_replicate3.sum()),
            all6counts_finite=int(np.isfinite(r[countcols]).all(axis=1).sum()),input_positive3=int((r[[f'count_e{i}_s0' for i in [1,2,3]]]>0).all(axis=1).sum()),
            **corrs,source_origin='Faure2022_reused' if domain=='psd95_pdz3' else 'MartiAranda2026'))
        for row,pos in zip(r.itertuples(),positions):
            scan=[helixmaps[domain][p] for p in pos if p in helixmaps[domain]]
            catalog.append(dict(source=str(info['path'].relative_to(ROOT)),domain=domain,gene=info['gene'],assay=info['assay'],phenotype=info['phenotype'],ligand=info['ligand'],block=info['block'],
                mutant=row.mutant,positions=';'.join(map(str,pos)),scan_sites=';'.join(map(str,scan)),Nham_aa=int(row.Nham_aa),row_index=int(row.row_index),input_count=float(row.input_count),
                finite_aggregate=bool(row.finite_aggregate),finite_replicate3=bool(row.finite_replicate3)))
        print('Audited',info['path'].name,'AA',len(r),flush=True)
    pd.DataFrame(file_stats).to_csv(OUT/'file_audit.csv',index=False);cat=pd.DataFrame(catalog);cat.to_csv(OUT/'identity_catalog.csv.gz',index=False,compression='gzip')
    # Pairing uses exact mutation identities; common nucleotide records are separately enumerated.
    pairing=[]
    for domain,ligands in LIGANDS.items():
        for ligand in ligands:
            ab=domain;bi=domain+'_'+ligand;aa_union=set();nt_aa_union=set();nt_count=0
            for block in ['b1','b2']:
                if (ab,block) not in variants:continue
                common=finite_sets[ab,block]&finite_sets[bi,block];aa_union|=common
                exact=ntsets[ab,block]&ntsets[bi,block];nt_count+=len(exact);nt_aa_union|={x[0] for x in exact}
            m=metadata[(metadata.pdz_name.str.lower()==domain)&(metadata.ligand_name.str.lower()==ligand)].iloc[0]
            pairing.append(dict(domain=domain,parent_gene=GENES[domain],ligand=ligand,ligand_sequence=m.ligand_sequence,
                ligand_sequence_standard20=all(a in AA for a in str(m.ligand_sequence)),
                matched_AA=len(aa_union),matched_single_AA=sum(':' not in a for a in aa_union),matched_double_AA=sum(':' in a for a in aa_union),
                matched_AA_with_common_nt=len(nt_aa_union),common_block_nt_records=nt_count,
                role='paired assay-specific abundance/binding supervision; not direct thermodynamic ground truth'))
    pd.DataFrame(pairing).to_csv(OUT/'paired_phenotypes.csv',index=False)
    multiligand=[]
    for domain,ligands in LIGANDS.items():
        if len(ligands)!=2:continue
        common=set()
        for block in ['b1','b2']:
            common |= finite_sets[domain,block]&finite_sets[domain+'_'+ligands[0],block]&finite_sets[domain+'_'+ligands[1],block]
        multiligand.append(dict(domain=domain,parent_gene=GENES[domain],ligands='|'.join(ligands),common_AA_three_readouts=len(common),singles=sum(':' not in v for v in common),doubles=sum(':' in v for v in common)))
    pd.DataFrame(multiligand).to_csv(OUT/'multi_ligand_overlap.csv',index=False)
    # Shared mutation patterns are conservatively treated as related, not new independent test examples.
    old=pd.read_csv(ROOT/'experiments/pdz2018_background_v1/historical_training_genotypes.csv.gz',usecols=['domain','left','right'])
    oldsets={d:set(zip(a.left,a.right)) for d,a in old.groupby('domain')};overlap=[]
    for domain,oldname in [('psd95_pdz2','pdz2'),('psd95_pdz3','pdz3')]:
        group=cat[(cat.domain==domain)&cat.finite_aggregate];allmut=set(group.mutant);mapped_s=set();mapped_d=set()
        for mutant in allmut:
            changes=mutant.split(':');pos=[int(m[1:-1]) for m in changes]
            if not all(p in helixmaps[domain] for p in pos):continue
            mapped=[m[0]+str(helixmaps[domain][p])+m[-1] for m,p in zip(changes,pos)]
            if len(mapped)==1:mapped_s.add(mapped[0])
            else:mapped_d.add(tuple(mapped))
        overlap.append(dict(aux_domain=domain,old_domain=oldname,shared_scan_single_identities=len(mapped_s),scan_only_double_identities=len(mapped_d),
            shared2018_double_patterns=len(mapped_d&oldsets[oldname]),interpretation='same9-site variant pattern; not proof of identical fullconstruct, phenotype or experiment'))
    pd.DataFrame(overlap).to_csv(OUT/'overlap_with2018.csv',index=False)
    eligible=cat[cat.finite_aggregate].copy();eligible['site_set']=eligible.scan_sites.fillna('').map(lambda s:set(map(int,s.split(';'))) if s else set())
    splits=[]
    for held in range(1,10):
        train=eligible[~eligible.site_set.map(lambda s:held in s)]
        assert not train.site_set.map(lambda s:held in s).any()
        splits.append(dict(regime='strict_aligned_site',held=str(held),training_rows=len(train),excluded_rows=len(eligible)-len(train),
            unique_domain_variant_pairs=len(train[['domain','mutant']].drop_duplicates()),label_overlap=0))
    for held in LIGANDS:
        train=eligible[eligible.domain!=held];assert (train.domain!=held).all()
        splits.append(dict(regime='leave_domain',held=held,training_rows=len(train),excluded_rows=len(eligible)-len(train),unique_domain_variant_pairs=len(train[['domain','mutant']].drop_duplicates()),label_overlap=0))
    for held in ['DLG4','PDZK1']:
        train=eligible[eligible.gene!=held];assert (train.gene!=held).all()
        splits.append(dict(regime='leave_parent_protein',held=held,training_rows=len(train),excluded_rows=len(eligible)-len(train),unique_domain_variant_pairs=len(train[['domain','mutant']].drop_duplicates()),label_overlap=0))
    pd.DataFrame(splits).to_csv(OUT/'split_eligibility.csv',index=False)
    metadata.to_csv(OUT/'development_assay_metadata.csv',index=False)
    check();dump(OUT/'execution.json',dict(status='completed',development_files=18,domains=4,parent_proteins=2,binding_conditions=6,abundance_conditions=4,
        identities=len(cat),new_model_fits=0,new_query_trials=0,heldout_outcome_files_read=0,author_fitted_weights_loaded=False))


if __name__=='__main__':
    {'freeze':freeze,'run':run}[sys.argv[1]]()
