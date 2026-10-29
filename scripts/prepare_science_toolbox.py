"""Prepare a fresh toolbox demonstration. Never rewrite an existing task or old run."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from science_toolbox import GROUPS, VERSION, ToolboxTask, save, serve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments' / VERSION
PRIVATE = ROOT / '.local' / VERSION
AA = 'ARNDCQEGHILKMFPSTWYV'
VOLUMES = json.loads((ROOT / 'data/public_reference/residue_volumes.json').read_text())


def structure_context(domain, mapping, distance):
    from prepare_structure_prior import AA3
    residues = {}
    for line in (ROOT/f"data/pdz2018/public/{mapping['pdb']}.pdb").read_text().splitlines():
        if line.startswith('ENDMDL'):
            break
        if not line.startswith('ATOM  ') or line[16] not in (' ', 'A') or line[17:20] not in AA3:
            continue
        atom = line[12:16].strip()
        if atom.startswith('H'):
            continue
        key = (line[21], line[22:27].strip())
        residue = residues.setdefault(key, dict(aa=AA3[line[17:20]], atoms={}))
        residue['atoms'].setdefault(atom, [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    helix = [(mapping['chain'], str(number)) for number in mapping['residue_numbers']]
    environments = []
    for site, key in enumerate(helix, 1):
        atoms = np.array(list(residues[key]['atoms'].values()))
        neighbors = []
        for other, residue in residues.items():
            if other == key:
                continue
            other_atoms = np.array(list(residue['atoms'].values()))
            closest = float(np.linalg.norm(atoms[:, None]-other_atoms[None, :], axis=2).min())
            if closest <= 4.5:
                neighbors.append(dict(chain=other[0], residue_number=other[1], aa=residue['aa'],
                    nearest_heavy_atom_angstrom=closest, outside_scanned_helix=other not in helix))
        environments.append(dict(scanned_site=site, pdb_residue=key[1], aa=residues[key]['aa'],
            nearby_residues=sorted(neighbors, key=lambda r:r['nearest_heavy_atom_angstrom'])))
    return dict(pdb=mapping['pdb'], chain=mapping['chain'], residue_numbers=mapping['residue_numbers'],
        distance=distance.tolist(), description=mapping['distance'], source_url=mapping['url'],
        reference_chain_sequence=''.join(r['aa'] for key, r in residues.items() if key[0] == mapping['chain']),
        reference_sequence_scope='PDB chain construct; not asserted identical to the assay construct',
        local_environments=environments, contact_cutoff_angstrom=4.5,
        limitations='WT heavy-atom contacts; not solvent accessibility, mutant stability, affinity or physical mechanism')


def public_candidates(sequence, damage, rows, distances):
    damage_site = int(damage[1:-1])-1
    return [dict(id=r['id'], site=int(r['id'][1:-1]), native=r['id'][0], mutant=r['id'][-1],
        group=next(k for k, group in GROUPS.items() if r['id'][-1] in group),
        WT_log_affinity=r['WT_log_affinity'], WT_affinity=math.exp(r['WT_log_affinity']),
        volume_change=VOLUMES[r['id'][-1]]-VOLUMES[r['id'][0]],
        distance_angstrom=float(distances[damage_site, int(r['id'][1:-1])-1])) for r in rows]


def synthetic_spec():
    sequence = 'HEDAVAALK'; damage = 'A4D'
    distance = np.abs(np.arange(9)[:, None]-np.arange(9)[None, :]) * 3.8
    rows = [dict(id=f'{native}{i+1}{mutant}', WT_log_affinity=-.02*AA.index(mutant))
        for i, native in enumerate(sequence) if i != 3 for mutant in AA if mutant != native]
    candidates = public_candidates(sequence, damage, rows, distance)
    labels = {r['id']: dict(affinity=math.exp(math.log(.1) + r['WT_log_affinity'] +
        (2.5 if r['group'] == 'hydrophobic' and r['distance_angstrom'] <= 8 else -.3))) for r in candidates}
    initial = sorted(candidates, key=lambda r: (-r['WT_log_affinity'], r['id']))[:8]
    return dict(task_id='synthetic_probe', kind='synthetic engineering demonstration', domain='synthetic',
        damage=damage, damage_affinity=.1, scanned_sequence=sequence,
        candidates=candidates, labels=labels, initial_ids=[r['id'] for r in initial], query_budget=32,
        structure=dict(pdb=None, residue_numbers=list(range(1, 10)), chain=None,
            distance=distance.tolist(), description='synthetic distances; not a biological structure'),
        evidence_boundary='synthetic labels; proves tool execution only, no scientific performance')


def development_specs():
    from pdz2018_data import DOMAINS, load_development
    # Only previously used development variables are requested. No sealed parent arrays.
    tasks = load_development()
    mappings = {r['domain']: r for r in json.loads((ROOT / 'experiments/pdz2018_point_dev_v1/structure_mapping.json').read_text())}
    chosen = []
    for domain in DOMAINS:
        ids = [k for k, t in tasks.items() if t['domain'] == domain and len(t['rows']) >= 32]
        key = min(ids, key=lambda k: hashlib.sha256((VERSION + '|' + k).encode()).hexdigest())
        t = tasks[key]; distance = np.load(ROOT / f'data/pdz2018/public/{domain}_geometry.npz')['distance']
        rows = t['rows']; candidates = public_candidates(DOMAINS[domain][0], t['damage'], rows, distance)
        mapping = mappings[domain]
        chosen.append(dict(task_id=key, kind='previously inspected PDZ development demonstration', domain=domain,
            damage=t['damage'], damage_affinity=math.exp(t['damage_log_affinity']),
            scanned_sequence=DOMAINS[domain][0], candidates=candidates,
            labels={r['id']: dict(affinity=r['affinity'], deltaG_kcal=r['deltaG_kcal']) for r in rows},
            initial_ids=[r['id'] for r in sorted(candidates, key=lambda r: (-r['WT_log_affinity'], r['id']))[:8]],
            query_budget=32, structure=structure_context(domain, mapping, distance),
            evidence_boundary='retrospective point-only development; no new wet-lab experiment, biological confidence or independent test'))
    return chosen


PROMPT = '''You are Protein Rescue Lab's scientific research agent. Keep the original damage and add one substitution at another assayed site. Seek measured relative affinity >=0.5 WT within32 total historical label queries, including8 already charged shared WT queries. This is a demonstration, not independent evaluation.
You have a scientific TOOLBOX, not a fixed research workflow. Choose analysis methods, candidate groups and experiments yourself. Full legal candidates and public reference geometry are available. Use science_state to recover, science_sequence/science_structure/science_candidates to inspect, science_predict for additive or observed-only ridge, science_python to write original analysis, science_record to propose/revise hypotheses, science_query for measured feedback, science_controls to check four controls, science_finish to commit a conclusion.
Python receives data, np and pd, with SciPy importable. Set JSON-serializable result. It cannot read project files, write files, access network or spawn subprocesses. Costs: at most12 Python attempts,80 computational mutations,32 total measured combinations. No arbitrary shell/file tools.
Record at least one testable hypothesis BEFORE new queries. Use the toolbox and your own Python analysis to investigate it. Commit1..4 selected IDs, a finite nonnegative predicted affinity per ID and a rationale BEFORE feedback. Revise at least one hypothesis after feedback, citing only observed IDs. Choose the next research step based on the evidence; do not repeat tools without a reason. Do not fit a complex model to a handful of points or claim reliable uncertainty.
You may stop after evidence supports a measured rescue and four controls have been checked, or exhaust the query budget, or finish with insufficient evidence. A few failures alone do not establish no possible rescue. Geometry, additive priors and fitted outputs are not measurements or causal mechanisms. Report limitations. Do not alter objectives or thresholds. Do not use unregistered tools, other sessions or known mutation answers. Actual native tool calls are required. End with science_finish.
'''


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--dataset', choices=['synthetic', 'pdz-dev'], default='synthetic')
    parser.add_argument('--synthetic-id', choices=['synthetic_probe', 'synthetic_python_probe'], default='synthetic_probe')
    parser.add_argument('--serve', action='store_true'); args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / 'protocol.json').exists():
        # Hash all existing scientific outputs and source files for non-destructive verification.
        prior = [p for folder in ['experiments', 'scripts', 'plugins'] for p in (ROOT / folder).rglob('*')
            if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts
            and 'science-toolbox' not in p.parts and p.name not in
            ['science_toolbox.py', 'toolbox_python.py', 'prepare_science_toolbox.py', 'test_science_toolbox.py',
             'toolbox_agh_client.mjs', 'install_science_toolbox.mjs', 'run_science_toolbox.mjs', 'report_science_toolbox.py', 'prepare_toolbox_runtime.py']]
        save(OUT / 'preserved_hashes.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in prior})
        save(OUT / 'protocol.json', dict(version=VERSION, scope='toolbox engineering + small retrospective development demonstration',
            query_budget=32, shared_initial=8, python_attempts=12, compute_calls=80,
            turn_timeout_ms=240000, max_completion_corrections=1,
            datasets=['synthetic', 'pdz-dev'], sealed_domains=['shank3', 'zo1'],
            cohort='one public-coverage-qualified identity-hash task per development domain; not selected by reachability',
            allowed_autonomy='analysis code, features, lightweight fitting, research order, legal candidate selection, stopping',
            fixed='damage, one-rescue constraint, point threshold, data access, budgets, evaluator',
            scientific_claim='no search improvement established by toolbox demonstration'))
        (OUT / 'prompt.txt').write_text(PROMPT)
    specs = [synthetic_spec()] if args.dataset == 'synthetic' else development_specs()
    if args.dataset == 'synthetic':
        specs[0]['task_id'] = args.synthetic_id
    tasks = []
    for spec in specs:
        task = ToolboxTask(PRIVATE / 'evaluator' / (spec['task_id'] + '.json'), spec); tasks.append(task)
        save(PRIVATE / 'public_tasks' / (spec['task_id'] + '.json'), task.public())
        snapshot = task.visible(); snapshot.pop('observed')
        save(OUT / 'public_tasks' / (spec['task_id'] + '.json'), snapshot)
    manifest_path = OUT / (args.dataset + '_manifest.json')
    old_ids = json.loads(manifest_path.read_text())['task_ids'] if manifest_path.exists() else []
    save(manifest_path, dict(task_ids=sorted(set(old_ids + [t.state['task_id'] for t in tasks])),
        dataset=args.dataset, sealed_domains_loaded=False, label_selection=False))
    if args.serve:
        serve(tasks, PRIVATE / 'oracle_endpoint.json')
    else:
        print(json.dumps(dict(prepared=[t.state['task_id'] for t in tasks], sealed_domains_loaded=False)))


if __name__ == '__main__':
    main()
