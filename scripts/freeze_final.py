"""Freeze final protocol and execution inputs before disclosing any heldout double."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
from experiment_data import ROOT


def main():
    dev=ROOT/'experiments/gb1_agnes_structure_dev_v1';out=ROOT/'experiments/gb1_final_holdout_v1'
    audit=json.loads((dev/'verification.json').read_text())
    assert audit['status']=='passed' and audit['complete_tasks']==audit['tasks']==34
    assert not (out/'freeze.json').exists() and not (ROOT/'.local'/out.name).exists()
    assert not (out/'fixed_policies/runs.csv').exists()
    p=json.loads((out/'protocol.json').read_text());selection=json.loads((ROOT/'experiments/gb1_factor_v2/selection.json').read_text())
    assert p['selected_method']==selection['selected_method']=='no_feedback_contact'
    paths=['scripts/factor_policy.py','scripts/structure_policy.py','scripts/adaptive_policy.py','scripts/menu_policy.py',
           'scripts/experiment_data.py','scripts/run_factor.py','scripts/run_pilot.py','scripts/agnes_stage_server.py','scripts/run_agnes_stage.mjs',
           'scripts/verify_factor.py','scripts/summarize_stage.py','scripts/report_three_steps.py','scripts/freeze_final.py',
           'data/public_reference/singles.csv','data/public_reference/position_mapping.csv','data/public_reference/structure_geometry.npz',
           'data/public_reference/residue_volumes.json','data/public_reference/1PGA.pdb','data/evaluator_only/gb1_raw.csv',
           'plugins/rescue-tools/index.mjs','plugins/rescue-tools/package.json',
           'vendor/agnes-harness/packages/sdk/src/index.node.ts','vendor/agnes-harness/packages/cli/dist/local/daemon.mjs',
           'vendor/agnes-harness/packages/cli/package.json','experiments/gb1_factor_v2/selection.json','experiments/gb1_factor_v2/protocol.json',
           'experiments/gb1_agnes_structure_dev_v1/verification.json','experiments/gb1_final_holdout_v1/protocol.json',
           'experiments/gb1_final_holdout_v1/fixed_policies/protocol.json','experiments/gb1_final_holdout_v1/prompt.txt']
    hashes={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'vendor/agnes-harness',text=True).strip()
    package=json.loads((ROOT/'experiments/agnes_package_inventory.json').read_text())['packages']
    plugin=next(x for x in package if x['id']=='@protein-rescue/query-tools')
    freeze=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes,official_AGH_commit=commit,
                plugin_integrity=plugin['integrity'],plugin_capability_hash=plugin['capabilityHash'],model=p['model'],endpoint=p['endpoint'],
                criterion='all 51 public-single-defined heldout backgrounds, no outcome-based exclusions; fixed selection and prompts unchanged after disclosure',
                corrections='at most two same-session budget-completion reminders, never reset state or exceed32',
                retrospective_boundary='public historical assay; pretraining knowledge cannot be excluded; no wet experiment')
    text=json.dumps(freeze,indent=2)+'\n';(out/'freeze.json').write_text(text);(out/'fixed_policies/freeze.json').write_text(text)
    print(json.dumps({k:freeze[k] for k in ['frozen_at_utc','official_AGH_commit','model']},indent=2));print('Frozen files:',len(hashes))


if __name__=='__main__':main()
