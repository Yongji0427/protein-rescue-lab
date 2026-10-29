"""Frozen v2 dominance experiment: pdz2018_gateless_v2 (development-only, point energy).

Three arms over all 207 PDZ-2018 tasks, seed0 deterministic. Because the proximity
exploration slot is ungated (proximity is always > 0), the trajectory is fully
determined under seed0 with no random fallback.

  wt_new    : plain top-32 public WT single-affinity rank (control; reproduces 80 old-success)
  ridge_dom : reserve top-24 public WT, fill remaining 8 slots with LCB of the FROZEN
              ridge residual model site_{s}_ridge.npz (rescue_ranking_v1)
  esm_dom   : reserve top-24 public WT, fill remaining 8 slots with LCB of the FROZEN
              ESM residual model site_{s}.npz (esm_v2)

Primary endpoint (gamma): net gain vs wt_new with 0 old-success loss.
Secondary disclosure (Q20-residual / P4): two-sided gate 3/2 recorded, NOT a hard gate.
Sealed shank3/zo1 never loaded. 0 Agnes calls in v2. Models loaded, not retrained (B1).
"""
import sys, json, hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pdz2018_data import ROOT, RT, DOMAINS, load_development, dump, hashes
from pdz_background_model import site, OnlinePredictor
from pdz_rescue_ranking import Model, THRESHOLD, public_task, features
from pdz_esm_model import ESMFeatures

OUT = ROOT / 'experiments/pdz2018_gateless_v2'
RIDGE_MODELS = ROOT / 'experiments/pdz2018_rescue_ranking_v1/models'
ESM_MODELS = ROOT / 'experiments/pdz2018_esm_v2/models'
ARMS = ['wt_new', 'ridge_dom', 'esm_dom']
RESERVED_N = 24
CAP = 32
RUNNER = 'run_pdz2018_gateless_v2.py'

MODEL_FILES = {
    'ridge': [f'site_{s}_ridge.npz' for s in range(9)],
    'esm': [f'site_{s}.npz' for s in range(9)],
}
DATA_FILES = ['data/pdz2018/source/Salinas_Ranganathan_data.mat',
              'data/public_reference/residue_volumes.json'] \
             + [f'data/pdz2018/public/{d}_geometry.npz' for d in DOMAINS]
DIAG_OUT = ['diagnosis.json', 'dominance_diagnosis.json', 'dominance_sweep.csv',
            'headroom_ledger.csv', 'headroom_detail.csv']


def base_for(t, arm, esm):
    s = site(t['damage'])
    public = public_task(t)
    if arm == 'ridge_dom':
        return Model.load(RIDGE_MODELS / f'site_{s}_ridge.npz').predict(features(public))
    return Model.load(ESM_MODELS / f'site_{s}.npz').predict(esm.task(public))


def trajectory(t, arm, esm):
    """Seed0 deterministic arm trajectory; returns queried candidate indices."""
    ids = np.array([r['id'] for r in t['rows']])
    wt = np.array([r['WT_log_affinity'] for r in t['rows']])
    energy = np.array([r['deltaG_kcal'] for r in t['rows']])
    additive = -RT * (t['damage_log_affinity'] + wt)
    order_wt = np.lexsort((ids, -wt))
    if arm == 'wt_new':
        return [int(i) for i in order_wt[:CAP]]
    seen = [int(i) for i in order_wt[:RESERVED_N]]
    pred = OnlinePredictor(t, base_for(t, arm, esm), True)
    pred.update(seen, energy[seen] - additive[seen])
    while len(seen) < CAP:
        b = pred.proposal(min(4, CAP - len(seen)), 'lcb')
        if not b:
            break
        seen.extend(b)
        pred.update(b, energy[b] - additive[b])
    return seen


def preserved_hashes():
    """Re-verify the ROOT-relative upstream freezes so the models/data we load are
    exactly what those experiments froze. Absolute ESM HF-cache paths are NOT
    reproducible on this box and are recorded as skipped, not silently dropped."""
    expected, skipped = {}, []
    for folder, fname in [('pdz2018_rescue_ranking_v1', 'freeze.json'),
                          ('pdz2018_background_v1', 'freeze.json'),
                          ('pdz2018_esm_v2', 'feature_freeze.json')]:
        z = json.loads((ROOT / 'experiments' / folder / fname).read_text())
        for k in ('hashes', 'model_hashes'):
            for key, digest in z.get(k, {}).items():
                if key.startswith('/'):
                    skipped.append(key)
                    continue
                expected[key] = digest
    for key, digest in expected.items():
        assert hashlib.sha256((ROOT / key).read_bytes()).hexdigest() == digest, key
    return expected, skipped


def freeze():
    assert not (OUT / 'freeze.json').exists(), 'Never replace a frozen experiment'
    OUT.mkdir(parents=True, exist_ok=True)
    preserved, skipped = preserved_hashes()
    dump(OUT / 'preserved_hashes.json', dict(preserved=preserved, skipped_abs_cache=skipped))
    protocol = dict(
        version=OUT.name,
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope='frozen development experiment; point energy only; NOT independent validation',
        domains=list(DOMAINS), sealed_unread=['shank3', 'zo1'], tasks=207,
        budget=32, batch=4, seed=0,
        endpoint='exp(-Gcombo/RT)>=0.5 WT; point only; all tasks retained',
        recipe=dict(
            dominance='reserve top-N=24 by public WT single-affinity rank, fill remaining 8 slots with LCB of the named frozen backfill model',
            reserved_N=RESERVED_N,
            n_star_source='diagnose_dominance.py: minimum reserved-WT count giving 0 old-success loss (both backfills N*=24)'),
        arms=[dict(name='wt_new', desc='control, plain top-32 public WT single-affinity rank'),
              dict(name='ridge_dom', desc='reserve top-24 WT + fill 8 with LCB of frozen ridge residual site_{s}_ridge.npz'),
              dict(name='esm_dom', desc='reserve top-24 WT + fill 8 with LCB of frozen ESM residual site_{s}.npz')],
        frozen_models=dict(
            ridge='experiments/pdz2018_rescue_ranking_v1/models/site_{0..8}_ridge.npz (loaded, not retrained)',
            esm='experiments/pdz2018_esm_v2/models/site_{0..8}.npz (loaded, not retrained)',
            esm_features='frozen ESM2-35M projection (esm_v2/feature_freeze.json)'),
        no_retrain='B1: residual models NOT retrained; only the search procedure changes (gate removed, N-reserve added)',
        gate_disclosure=dict(
            type='pre-registered secondary disclosure endpoint (gamma/P4); NOT a hard pass/fail',
            two_sided_gate=dict(headroom_gain_at_least=3, old_success_loss_at_most=2),
            anchor='K1 diagnostic reachable-headroom count (diagnosis.json); disclosure number frozen here, no post-hoc relaxation'),
        primary_endpoint='net gain in full-207 successes vs wt_new baseline (80 old-success) with 0 old-success loss',
        agreement='seed0 deterministic trajectory == dominance_sweep.csv N=24 row (the diagnostic is the frozen experiment)',
        agnes='0 Agnes calls in v2; sealed shank3/zo1 remain unread; v3 uses the winning arm',
        interpretation='exploratory development; dominance vs net-0 framing; no biological confidence or independent validation claims',
        method_sources=['https://gaussianprocess.org/gpml/chapters/RW2.pdf',
                       'https://scikit-learn.org/stable/modules/linear_model.html'])
    dump(OUT / 'protocol.json', protocol)
    paths = [ROOT / 'scripts' / n for n in [RUNNER, 'diagnose_dominance.py', 'diagnose_gateless_headroom.py',
                                            'pdz_background_model.py', 'pdz2018_data.py',
                                            'pdz_rescue_ranking.py', 'pdz_esm_model.py', 'pdz2018_policy.py']]
    paths += [OUT / 'protocol.json', OUT / 'preserved_hashes.json']
    paths += [RIDGE_MODELS / f for f in MODEL_FILES['ridge']]
    paths += [ESM_MODELS / f for f in MODEL_FILES['esm']]
    for d in DATA_FILES:
        paths += [ROOT / d]
    for d in DIAG_OUT:
        paths += [OUT / d]
    dump(OUT / 'freeze.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
                                   hashes=hashes(paths), reserved_N=RESERVED_N, seed=0, arms=ARMS,
                                   boundary='parameters fixed before this new dominance search; development labels previously audited; ESM HF-cache model_hashes not reproducible on this box, recorded in preserved_hashes.json'))
    print('Frozen gateless_v2 dominance protocol', flush=True)


def run():
    assert not (OUT / 'result.json').exists(), 'Frozen output cannot be rerun/overwritten'
    f = json.loads((OUT / 'freeze.json').read_text())
    for k, v in f['hashes'].items():
        assert hashlib.sha256((ROOT / k).read_bytes()).hexdigest() == v, k
    tasks = load_development()
    assert len(tasks) == 207
    esm = ESMFeatures()
    runs, qlog = [], []
    for tid, t in tasks.items():
        rows = t['rows']; ids = np.array([r['id'] for r in rows])
        aff = np.array([r['affinity'] for r in rows])
        for arm in ARMS:
            seen = trajectory(t, arm, esm)
            best = max(aff[i] for i in seen)
            runs.append(dict(task_id=tid, domain=t['domain'], damage=t['damage'],
                            damage_site=site(t['damage']) + 1, method=arm,
                            success=bool(best >= .5), best_affinity=float(best), queries=len(seen)))
            for q, i in enumerate(seen, 1):
                qlog.append(dict(task_id=tid, method=arm, query=q, rescue=ids[i], affinity=rows[i]['affinity']))
    pd.DataFrame(runs).to_csv(OUT / 'runs.csv', index=False)
    pd.DataFrame(qlog).to_csv(OUT / 'query_log.csv.gz', index=False, compression='gzip')

    ledger = pd.read_csv(OUT / 'headroom_ledger.csv').set_index('task_id')
    headroom_ids = list(ledger.query('category=="headroom"').index)
    wt_success_ids = list(ledger.query('category=="wt_success"').index)
    runs = pd.DataFrame(runs)
    wt_succ = runs[runs.method == 'wt_new'].set_index('task_id')['success']
    per_arm = {}
    for arm in ARMS:
        s = runs[runs.method == arm].set_index('task_id')['success']
        h_gained = int(s.loc[headroom_ids].sum())
        w_lost = int((~s.loc[wt_success_ids]).sum())
        total = int(s.sum())
        per_arm[arm] = dict(successes=total, headroom_gained=h_gained, old_success_lost=w_lost,
                            net_vs_wt_new=total - int(wt_succ.sum()),
                            two_sided_gate_pass=bool(h_gained >= 3 and w_lost <= 2))
    dump(OUT / 'result.json', dict(
        per_arm=per_arm,
        primary='net gain vs wt_new baseline (80 old-success) with 0 old-success loss',
        gate_disclosure='two-sided gate 3/2 recorded as pre-registered secondary disclosure (gamma/P4); NOT a hard gate',
        winner_by_primary=max(ARMS, key=lambda a: per_arm[a]['net_vs_wt_new']),
        boundary='development only; sealed shank3/zo1 unread; 0 Agnes calls; point energy; seed0 deterministic',
        evidence='dominance_sweep.csv N=24 row reproduced; not independent confirmation'))
    print(json.dumps(json.loads((OUT / 'result.json').read_text()), indent=2), flush=True)


def verify():
    f = json.loads((OUT / 'freeze.json').read_text())
    for k, v in f['hashes'].items():
        assert hashlib.sha256((ROOT / k).read_bytes()).hexdigest() == v, k
    print('all frozen hashes match', len(f['hashes']), 'files', flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'freeze':
        freeze()
    elif cmd == 'run':
        run()
    elif cmd == 'verify':
        verify()
    else:
        raise ValueError('Use freeze | run | verify')
