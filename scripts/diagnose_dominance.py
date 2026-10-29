"""R3=c diagnostic: minimum reserved-WT count N* for the dominance arm.

The dominance arm (R2=1) reserves the top-N candidates by public WT single-affinity
rank (protecting old successes), then fills the remaining (32-N) slots with a frozen
residual LCB model (ridge or ESM) to capture headroom. Goal: "only increase, never
decrease" -> find the smallest N with ZERO old-success loss, and report headroom
rescues at that N.

This sweeps N=1..32 for both backfill models over the 98 reachable backgrounds
(18 headroom + 80 wt_success). Deterministic seed0; no sealed labels.
"""
import json
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
BACKFILLS = ['ridge', 'esm']
CAP = 32


def categorize(tasks):
    headroom, wt_success = {}, {}
    for task_id, t in tasks.items():
        ids = np.array([r['id'] for r in t['rows']])
        wt = np.array([r['WT_log_affinity'] for r in t['rows']])
        energy = np.array([r['deltaG_kcal'] for r in t['rows']])
        order_wt = np.lexsort((ids, -wt))
        true_set = frozenset(i for i in range(len(ids)) if energy[i] <= THRESHOLD)
        if not true_set:
            continue
        rec = dict(t=t, ids=ids, wt=wt, energy=energy, order_wt=order_wt,
                   true_set=true_set,
                   additive=-RT * (t['damage_log_affinity'] + wt))
        if true_set & set(int(i) for i in order_wt[:CAP]):
            wt_success[task_id] = rec
        else:
            headroom[task_id] = rec
    return headroom, wt_success


def dominance_seen(rec, backfill, N, esm_store):
    """Reserve top-N by WT rank, then fill remaining slots with LCB backfill.
    Returns the set of queried candidate indices.
    A FRESH predictor is built per call (state must not leak across N values)."""
    ids, wt, energy, order_wt, additive, t = (rec['ids'], rec['wt'], rec['energy'],
                                              rec['order_wt'], rec['additive'], rec['t'])
    public = public_task(t)
    s = site(t['damage'])
    if backfill == 'ridge':
        base = Model.load(RIDGE_MODELS / f'site_{s}_ridge.npz').predict(features(public))
    else:
        base = Model.load(ESM_MODELS / f'site_{s}.npz').predict(esm_store.task(public))
    predictor = OnlinePredictor(t, base, True)
    seen = [int(i) for i in order_wt[:N]]
    if seen:
        predictor.update(seen, energy[seen] - additive[seen])
    while len(seen) < CAP:
        size = min(4, CAP - len(seen))
        batch = predictor.proposal(size, 'lcb')
        if not batch:
            break
        seen.extend(batch)
        predictor.update(batch, energy[batch] - additive[batch])
    return set(seen)


def main():
    tasks = load_development()
    assert len(tasks) == 207
    headroom, wt_success = categorize(tasks)
    old = pd.read_csv(OLD_MISSED)
    assert set(old['task_id']) == set(headroom), 'headroom drift'
    assert len(headroom) == 18 and len(wt_success) == 80, (len(headroom), len(wt_success))
    esm_store = ESMFeatures()

    rows = []
    for backfill in BACKFILLS:
        for N in range(1, CAP + 1):
            h_rescued, w_lost = 0, 0
            for tid, rec in headroom.items():
                if dominance_seen(rec, backfill, N, esm_store) & rec['true_set']:
                    h_rescued += 1
            for tid, rec in wt_success.items():
                if not (dominance_seen(rec, backfill, N, esm_store) & rec['true_set']):
                    w_lost += 1
            rows.append(dict(backfill=backfill, N=N, headroom_rescued=h_rescued,
                            wt_success_lost=w_lost,
                            zero_loss=int(w_lost == 0)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'dominance_sweep.csv', index=False)

    report = {}
    for backfill in BACKFILLS:
        sub = df[df.backfill == backfill]
        zero = sub[sub.zero_loss == 1]
        N_star = int(zero['N'].min()) if len(zero) else None
        at = sub[sub.N == N_star].iloc[0] if N_star is not None else sub.iloc[0]
        report[backfill] = dict(
            N_star_zero_loss=N_star,
            headroom_rescued_at_N_star=int(at.headroom_rescued) if N_star is not None else None,
            wt_success_lost_at_N_star=int(at.wt_success_lost) if N_star is not None else None,
            max_headroom_any_N_with_zero_loss=int(zero['headroom_rescued'].max()) if len(zero) else 0,
            n_values_with_zero_loss=int(len(zero)))
    # Context: N=0 analogue (all 32 slots model, no reserved WT) for the gain/loss trade-off.
    dump(OUT / 'dominance_diagnosis.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='R3=c: smallest reserved-WT count N* giving zero old-success loss for the dominance arm',
        rule='reserve top-N by public WT rank, fill remaining with frozen LCB backfill; deterministic seed0',
        headroom=18, wt_success=80,
        per_backfill=report,
        boundary='development labels only; sealed shank3/zo1 not loaded'))
    print('=== dominance sweep: headroom_rescued / wt_success_lost per N ===')
    for backfill in BACKFILLS:
        sub = df[df.backfill == backfill]
        nstar = report[backfill]['N_star_zero_loss']
        print(f'--- backfill={backfill}  N*= {nstar} '
              f'(headroom@N*={report[backfill]["headroom_rescued_at_N_star"]}, '
              f'zero-loss N values={report[backfill]["n_values_with_zero_loss"]})')
        show = sub[sub.N.isin([1, 8, 16, 24, 32] + ([nstar] if nstar is not None else []))]
        print(show.to_string(index=False))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
