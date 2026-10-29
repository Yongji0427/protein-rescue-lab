"""K1 precise-slot diagnostic for pdz2018_gateless_v2; development labels only.

Simulates the three v2 arms (deterministic seed0, M1 batch recipe, ungated
proximity exploration slot) over ALL 98 reachable PDZ-2018 backgrounds:
  - 18 headroom backgrounds  (true rescue NOT at WT single-affinity rank<=32) -> gains side
  - 80 wt_success backgrounds (true rescue at WT rank<=32, found by old WT)    -> losses side

Reports, per arm: how many headroom it rescues and how many wt_success it
DROPS. This is the complete input to the frozen two-sided gate (Q18):
  GATE = rescue >= 3 of 18 headroom  AND  lose <= 2 of 80 wt_success.
The gate gain number (3) is an INDEPENDENT constant chosen by the user, NOT
derived from arm output; capturable counts are context only.

No sealed SHANK3/ZO1 labels are loaded; load_development() asserts that.
"""
import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pdz2018_data import ROOT, DOMAINS, RT, load_development, dump
from pdz_background_model import site, OnlinePredictor
from pdz_rescue_ranking import Model, THRESHOLD, public_task, features
from pdz_esm_model import ESMFeatures

OUT = ROOT / 'experiments/pdz2018_gateless_v2'
RIDGE_MODELS = ROOT / 'experiments/pdz2018_rescue_ranking_v1/models'
ESM_MODELS = ROOT / 'experiments/pdz2018_esm_v2/models'
OLD_MISSED = ROOT / 'experiments/pdz2018_background_v1/missed_backgrounds.csv'
ARMS = ['wt_new', 'ridge_gateless', 'esm_gateless']
SIGMA = 6.0
GATE_GAIN = 3      # Q18: independent constant, NOT derived from arm output
GATE_LOSS_CAP = 2  # Q11/Q8: lose <= 2 of the 80 old WT successes


def seed_for(task):
    # Same SeedSequence recipe as production runs. The ungated proximity score
    # is strictly > 0 for every candidate, so the exploration slot never falls
    # back to a random draw and the whole trajectory is deterministic.
    return np.random.SeedSequence([20261008, list(DOMAINS).index(task['domain']),
                                   site(task['damage']) + 1, ord(task['damage'][-1]), 0])


def ungated_contact(task):
    """exp(-(d/6)^2) with the eligible chemical gate REMOVED (Q5 option A)."""
    distance = np.load(ROOT / f"data/pdz2018/public/{task['domain']}_geometry.npz")['distance']
    d = site(task['damage'])
    return np.exp(-((distance[d, [site(r['id']) for r in task['rows']]]) / SIGMA) ** 2)


def simulate(task, arm, contact, ids, wt, energy, additive, order_wt, predictor):
    """M1 recipe: shared initial 8 (WT top-8, identical across arms), then up to
    6 batches of (3 model slots + 1 ungated proximity exploration slot).
    Partial final batch: arm's model slots only. Batch committed before labels.
    Returns (queried indices in order, list of batch strings, per-query slot role).
    Slot roles: 'initial_wt' (queries 0-7), 'model@b{k}' / 'explore@b{k}' where
    k is the 1-based batch number (k=3..8); partial final batch: model slots only.
    """
    cap = min(32, len(ids))
    seen, batches, roles = [], [], []
    batch_no = 0
    while len(seen) < cap:
        size = min(4, cap - len(seen))
        remaining = [int(i) for i in order_wt if i not in seen]
        if len(seen) < 8:
            batch = remaining[:size]
            role = 'initial_wt'
        elif size >= 4:
            batch_no += 1
            k = 2 + batch_no  # initial 8 = batches 1-2; exploration batches are 3..8
            model_slots = (predictor.proposal(3, 'lcb') if arm != 'wt_new'
                           else remaining[:3])
            taken = set(model_slots)
            cand = np.array([i for i in remaining if i not in taken])
            c, w = contact[cand], wt[cand]
            rank = np.lexsort((ids[cand].astype(str), -w, -c))
            batch = model_slots + [int(cand[rank[0]])]
            role = 'model'
        else:  # partial final batch: arm's model/WT slots only, no exploration slot
            batch = (predictor.proposal(size, 'lcb') if arm != 'wt_new'
                     else remaining[:size])
            role = 'model'
        assert len(batch) == size and not set(batch) & set(seen)
        seen.extend(batch)
        if role == 'initial_wt':
            roles.extend(['initial_wt'] * size)
        elif size >= 4:  # 3 model slots + 1 exploration slot, batch number k
            roles.extend([f'model@b{k}'] * 3 + [f'explore@b{k}'])
        else:  # partial final batch: model slots only, no exploration slot
            batch_no += 1
            k = 2 + batch_no
            roles.extend([f'model@b{k}'] * size)
        batches.append([str(ids[i]) for i in batch])
        if predictor is not None:
            predictor.update(batch, energy[batch] - additive[batch])
    return seen, batches, roles


def main():
    tasks = load_development()
    assert len(tasks) == 207
    esm_store = ESMFeatures()

    headroom, wt_success, arrays = [], [], {}
    for task_id, t in tasks.items():
        ids = np.array([r['id'] for r in t['rows']])
        wt = np.array([r['WT_log_affinity'] for r in t['rows']])
        energy = np.array([r['deltaG_kcal'] for r in t['rows']])
        order_wt = np.lexsort((ids, -wt))
        true_set = frozenset(i for i in range(len(ids)) if energy[i] <= THRESHOLD)
        if not true_set:
            continue  # unreachable: stays in the denominator, not simulated
        arrays[task_id] = dict(ids=ids, wt=wt, energy=energy, order_wt=order_wt,
                               true_set=true_set, task=t)
        if true_set & set(int(i) for i in order_wt[:min(32, len(ids))]):
            wt_success.append(task_id)
        else:
            headroom.append(task_id)
    headroom_set = set(headroom)
    old = pd.read_csv(OLD_MISSED)
    assert set(old['task_id']) == headroom_set, 'headroom set drifted from the frozen 18'
    assert len(headroom) == 18 and len(wt_success) == 80, (len(headroom), len(wt_success))

    ledger, detail = [], []
    for task_id in sorted(headroom_set | set(wt_success)):
        a = arrays[task_id]
        t = a['task']
        ids, wt, energy, order_wt, true_set = a['ids'], a['wt'], a['energy'], a['order_wt'], a['true_set']
        additive = -RT * (t['damage_log_affinity'] + wt)
        contact = ungated_contact(t)
        public = public_task(t)
        s = site(t['damage'])
        prox_order = np.lexsort((ids, -contact))
        prox_rank = {int(i): r + 1 for r, i in enumerate(prox_order)}
        arms_seen, captured, arm_role = {}, {}, {}
        for arm in ARMS:
            predictor = None
            if arm == 'ridge_gateless':
                base = Model.load(RIDGE_MODELS / f'site_{s}_ridge.npz').predict(features(public))
                predictor = OnlinePredictor(t, base, True)
            elif arm == 'esm_gateless':
                base = Model.load(ESM_MODELS / f'site_{s}.npz').predict(esm_store.task(public))
                predictor = OnlinePredictor(t, base, True)
            seen, _, roles = simulate(t, arm, contact, ids, wt, energy, additive, order_wt, predictor)
            arms_seen[arm] = set(seen)
            arm_role[arm] = dict(zip(seen, roles))
            captured[arm] = sorted(str(ids[i]) for i in true_set & arms_seen[arm])
        row = dict(task_id=task_id, domain=t['domain'], damage=t['damage'], damage_site=s + 1,
                   category='headroom' if task_id in headroom_set else 'wt_success',
                   n_candidates=len(ids), n_true=len(true_set),
                   true_min_prox_rank=min(prox_rank[i] for i in true_set),
                   n_true_prox_le8=sum(prox_rank[i] <= 8 for i in true_set))
        for arm in ARMS:
            row[f'{arm}_captures'] = ';'.join(captured[arm])
            row[f'{arm}_n'] = len(captured[arm])
        row['capturable'] = any(captured[arm] for arm in ARMS)
        ledger.append(row)
        if task_id in headroom_set:  # per-candidate detail stays focused on the 18
            for i in sorted(true_set):
                captured = ';'.join(f'{arm}:{arm_role[arm][i]}'
                                    for arm in ARMS if i in arms_seen[arm])
                detail.append(dict(task_id=task_id, rescue=str(ids[i]), energy_kcal=float(energy[i]),
                                   combo_affinity=float(np.exp(-energy[i] / RT)),
                                   wt_rank=int(np.flatnonzero(order_wt == i)[0]) + 1,
                                   prox_rank=prox_rank[i], prox_le8=int(prox_rank[i] <= 8),
                                   captured_by=captured))
    df = pd.DataFrame(ledger)
    h = df[df.category == 'headroom']
    w = df[df.category == 'wt_success']
    headroom_rescued = {arm: int((h[f'{arm}_n'] > 0).sum()) for arm in ARMS}
    wt_success_lost = {arm: int((w[f'{arm}_n'] == 0).sum()) for arm in ARMS}
    n_capturable = int(h['capturable'].sum())
    n_oracle_in_budget = int((h['true_min_prox_rank'] <= 32).sum())
    arm_passes = {arm: (headroom_rescued[arm] >= GATE_GAIN and wt_success_lost[arm] <= GATE_LOSS_CAP)
                  for arm in ARMS}

    pd.DataFrame(ledger).to_csv(OUT / 'headroom_ledger.csv', index=False)
    pd.DataFrame(detail).to_csv(OUT / 'headroom_detail.csv', index=False)
    # Data-driven slot attribution across the captured headroom candidates.
    slot_tally = {'initial_wt': 0, 'model': 0, 'explore': 0}
    n_captured_candidates = 0
    for d in detail:
        cb = d['captured_by']
        if not cb:
            continue
        n_captured_candidates += 1
        for part in cb.split(';'):
            role = part.rsplit(':', 1)[-1]           # e.g. model@b4 / explore@b5 / initial_wt
            base = 'initial_wt' if role.startswith('initial_wt') else role.split('@', 1)[0]
            slot_tally[base] = slot_tally.get(base, 0) + 1
    arm_caps = {arm: int(sum(1 for r in h[f'{arm}_n'] if r > 0)) for arm in ARMS}
    dump(OUT / 'diagnosis.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope='K1 precise-slot pre-analysis over all 98 reachable PDZ-2018 backgrounds; development labels only',
        n_headroom=18, n_wt_success=80, headroom_task_ids=sorted(headroom_set),
        arms=ARMS, sigma=SIGMA, chemical_gate_removed=True,
        batch_recipe='shared initial8 WT; then 6x (3 model slots + 1 ungated proximity exploration slot); partial final batch model slots only',
        model_sources=dict(ridge='frozen site-fold ridge from pdz2018_rescue_ranking_v1/models (no retrain)',
                          esm='frozen ESM2-35M-augmented ridge from pdz2018_esm_v2/models + frozen embeddings (zero ESM recomputation)'),
        seed='SeedSequence([20261008, domain_idx, site+1, ord(damage_aa), 0]); deterministic, no random draw fires',
        # --- HONEST context numbers (development evidence, NOT gate inputs) ---
        n_capturable_any_arm=n_capturable,
        n_headroom_oracle_in_budget_le32=n_oracle_in_budget,
        k1_note=f'data-driven: across the {n_captured_candidates} captured headroom candidate captures, slot attribution = {slot_tally}; ' +
                 ('the ungated exploration slot does NOT drive the gain (0 explore captures).' if slot_tally['explore'] == 0 else
                  f'the exploration slot DID capture {slot_tally["explore"]} headroom candidate(s), so the gate removal is a partial driver; see headroom_detail.csv captured_by for the slot-level truth.'),
        headroom_per_arm_captured=arm_caps,
        # --- gate ---
        gate_rule=f'independent two-sided gate: rescue >= {GATE_GAIN} of the 18 headroom AND lose <= {GATE_LOSS_CAP} of the 80 wt_success; G2 = any arm passes',
        gate_gain_constant=GATE_GAIN, gate_loss_cap=GATE_LOSS_CAP,
        gate_gain_provenance='independent constant (Q18), consistent with project historical net>=3; NOT derived from n_capturable',
        per_arm_predicted=dict(headroom_rescued=headroom_rescued, wt_success_lost=wt_success_lost,
                               passes_g2=arm_passes),
        narrative_correction='Q19 K+M: improvement driver = residual LCB slots, not the gate removal; v2 framed as "gateless + three-arm residual search".',
        boundary='development labels only; sealed shank3/zo1 not loaded'))
    print(f'headroom=18 wt_success=80 capturable_any_arm={n_capturable} oracle_in_budget(<=32)={n_oracle_in_budget} gate_gain={GATE_GAIN} loss_cap={GATE_LOSS_CAP}')
    for arm in ARMS:
        print(f'  {arm:14s} headroom_rescued={headroom_rescued[arm]:2d}  wt_success_lost={wt_success_lost[arm]:2d}  passes_g2={arm_passes[arm]}')
    cols = ['task_id', 'category', 'n_true', 'true_min_prox_rank'] + [f'{a}_n' for a in ARMS] + ['capturable']
    print(pd.DataFrame(ledger)[[c for c in cols if c in ledger[0].keys()]].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
