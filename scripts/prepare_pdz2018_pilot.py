"""Freeze a point-only development pilot; sealed parent proteins remain unread."""
from datetime import datetime, timezone
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.io import loadmat
from audit_pdz2018 import fit_calibration
from prepare_structure_prior import AA3
from pdz2018_data import ROOT, OUT, DOMAINS, dump, hashes, load_development


def main():
    assert not (OUT/'freeze.json').exists(), 'Do not overwrite frozen pilot'
    OUT.mkdir(parents=True,exist_ok=True)
    mapping = []
    for domain,(sequence,code) in DOMAINS.items():
        residues = {}
        for line in (ROOT/f'data/pdz2018/public/{code}.pdb').read_text().splitlines():
            if line.startswith('ENDMDL'): break
            if not line.startswith('ATOM  ') or line[16] not in [' ','A']: continue
            key=(line[21],line[22:27])
            residue=residues.setdefault(key,dict(aa=AA3[line[17:20]],atoms={}))
            residue['atoms'].setdefault(line[12:16].strip(),np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])]))
        matches=[]
        for chain in sorted({k[0] for k in residues}):
            rows=[(k,v) for k,v in residues.items() if k[0]==chain]
            seq=''.join(v['aa'] for k,v in rows)
            for i in range(len(seq)-8):
                if seq[i:i+9]==sequence: matches.append(rows[i:i+9])
        assert matches, (domain,sequence,'No exact public structure helix match')
        rows=matches[0];xyz=[]
        for key,v in rows:
            side=[coord for atom,coord in v['atoms'].items() if atom not in ['N','CA','C','O','OXT'] and not atom.startswith('H')]
            xyz.append(np.array(side or [v['atoms']['CA']]))
        distance=np.array([[np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2).min() for b in xyz] for a in xyz])
        assert np.allclose(distance,distance.T)
        np.savez(ROOT/f'data/pdz2018/public/{domain}_geometry.npz',distance=distance)
        mapping.append(dict(domain=domain,pdb=code,url=f'https://files.rcsb.org/download/{code}.pdb',helix=sequence,
            chain=rows[0][0][0],residue_numbers=[key[1].strip() for key,v in rows],exact_matches=len(matches),
            distance='minimum WT side-chain heavy-atom distance; glycine CA; first model, first exact matching chain'))
    dump(OUT/'structure_mapping.json',mapping)
    source=loadmat(ROOT/'data/pdz2018/source/Salinas_Ranganathan_data.mat',variable_names=list(DOMAINS),squeeze_me=True,struct_as_record=False)
    diagnostic=[]
    for domain in DOMAINS:
        o=source[domain];theta,rmse,energy=fit_calibration(o)
        x=np.log(o.selected_single_counts/o.unselected_single_counts/(o.selected_wt_counts/o.unselected_wt_counts));mask=np.isfinite(x)&np.isfinite(o.deltaG_single)
        diagnostic.append(dict(domain=domain,single_curve_RMSE_kcal=rmse,
            raw_enrichment_energy_rank_correlation=float(spearmanr(x[mask],o.deltaG_single[mask]).statistic),
            exact_monotone_mapping_supported=rmse<=.001,alpha=float(theta[0]),fraction_WT=float(theta[1])))
    dump(OUT/'endpoint_diagnosis.json',dict(diagnostics=diagnostic,
        conclusion='PDZ3 mapping remains unresolved; published point energies supported; do not synthesize per-variant confidence intervals',
        evidence='paper reports four PDZ3 replicate experiments but deposited count fields are single arrays; cannot establish replicate-to-energy mapping from available fields',
        primary='exp(-published WT-relative deltaG/RT)>=0.5',
        secondary='two-domain conditional count sensitivity only; unavailable PDZ3 values stay null',
        sealed_domains_loaded=False))
    tasks=load_development()
    manifest=pd.DataFrame([dict(task_id=k,domain=t['domain'],damage=t['damage'],damage_affinity=float(np.exp(t['damage_log_affinity'])),candidates=len(t['rows'])) for k,t in tasks.items()])
    assert len(tasks)==207
    old=pd.read_csv(ROOT/'experiments/dataset_feasibility_v1/pdz2018_tasks.csv').query("regime=='all_R'")
    expected={(r.domain,r.damage):r.candidates for r in old.itertuples()}
    assert all(expected[t['domain'],t['damage']]==len(t['rows']) for t in tasks.values())
    manifest.to_csv(OUT/'task_manifest.csv',index=False)
    cohort=[]
    for domain,n in [('pdz2',11),('pdz3',11),('syntrophin',10)]:
        ids=[k for k,t in tasks.items() if t['domain']==domain and len(t['rows'])>32]
        cohort+=sorted(ids,key=lambda k:hashlib.sha256(('pdz2018_point_dev_v1|'+k).encode()).hexdigest())[:n]
    p=dict(version=OUT.name,status='point-only development pilot; not formal reliable-rescue benchmark',
        damage_tasks=cohort,all_damage_tasks=list(tasks),split='development',unread_test_domains=['shank3','zo1'],
        cohort_selection='identity SHA256 per domain 11/11/10; only public candidate coverage >32; no reachability enrichment',
        seed=20261008,repeats=100,budget=32,batch_size=4,initial_WT_queries=8,
        short_pool_policy='retain all tasks in full fixed evaluation, exhaust min(32,N); last batch may be short',
        primary='published double deltaG <= RT*log(2); equivalently relative association affinity >=0.5WT',
        secondary='pdz2/syntrophin fixed-calibration conditional count sensitivity; PDZ3 null; no aggregate reliable claim',
        selected_method='no_feedback_contact',methods=['wt_rank','no_feedback_contact'],
        structure_policy='unchanged GB1 neutral/hydrophobic gate and exp(-(distance/6)^2), minimum WT side-chain distance; top3 WT + one proximity slot, uniform fallback',
        model='agnes-3.0-flash',endpoint='https://api.agnes-ai.cn/v1',max_completion_corrections=2,turn_timeout_ms=180000,
        evidence='all development outcomes previously inspected in feasibility audit; no independent validation; no wet-lab or causal mechanism claims',
        formal_holdout_gate='unresolved reliable endpoint; no test labels read or model trials on test proteins')
    dump(OUT/'protocol.json',p)
    (OUT/'prompt.txt').write_text('''You are Protein Rescue Lab using retrospective experimental PDZ binding energies. Keep the damage mutation and add exactly one rescue at another of the nine scanned alpha2-helix sites. Find measured relative association affinity >=0.5 WT within 32 unique queries. This is a POINT-ONLY development pilot. count_lower_affinity is optional conditional-count sensitivity, unavailable for PDZ3, not biological confidence.
Eight shared WT-ranked labels are already queried and charged. Complete SIX real query_pdz_batch calls, four distinct CURRENT menu candidates per call including required_WT_anchor, to reach 32. Commit all four before feedback. Do not stop early after success. Use ONLY query_pdz_batch and the supplied public states. Do not read files, use shell/web, other tools, other tasks or known mutation answers. Ordinary tool-like text does not execute.
WT_log_affinity is a public single-mutant measurement. estimated_combo_log_affinity is the additive public single prior, not an unqueried double measurement. structure_score is a WT distance heuristic with a fixed chemical gate, not mutant stability or mechanism. The fixed_rule_proposal is a suggestion; choose the other three using observed feedback and public priors. Record rationale <=100 characters.
The tool returns observations and the next menu. When done=true, stop and report actual measurements briefly. Correct invalid calls without extra budget. Do not invent results or claim experimental validation.
''')
    paths=[ROOT/'scripts'/name for name in ['pdz2018_data.py','pdz2018_policy.py','prepare_pdz2018_pilot.py','run_pdz2018_fixed.py','pdz2018_oracle_server.py','run_pdz2018_agnes.mjs']]
    paths += [OUT/'protocol.json',OUT/'prompt.txt',OUT/'task_manifest.csv',OUT/'structure_mapping.json',
        ROOT/'data/pdz2018/source/Salinas_Ranganathan_data.mat',ROOT/'experiments/dataset_feasibility_v1/pdz2018_calibration_QC.csv']
    paths += list((ROOT/'data/pdz2018/public').glob('*'))
    paths += list((ROOT/'plugins/pdz-tools').glob('*'))
    dump(OUT/'freeze.json',dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes(paths),
        boundary='frozen before new search trajectories/model calls; development labels already audited; test arrays unread'))
    print('Frozen development pilot:',len(tasks),'tasks;',len(cohort),'model tasks')


if __name__=='__main__': main()
