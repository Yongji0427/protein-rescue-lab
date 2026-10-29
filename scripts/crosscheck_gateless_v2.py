"""Read-only cross-check: does the FROZEN runner reproduce the FROZEN diagnostic?

Compares, for every reachable task at RESERVED_N=24 (seed0), the exact queried
candidate set produced by:
  - run_pdz2018_gateless_v2.trajectory  (the runner that will be frozen/executed)
  - diagnose_dominance.dominance_seen   (the diagnostic that produced dominance_sweep.csv)

If every per-task set is identical, the runner's aggregate MUST equal the N=24
row of dominance_sweep.csv. This verifies the "agreement" field of protocol.json
is a fact, not a design claim. Writes nothing; no run(), no result.json.
"""
import json
import pandas as pd
import run_pdz2018_gateless_v2 as run_mod
import diagnose_dominance as diag_mod
from pdz2018_data import load_development
from pdz_esm_model import ESMFeatures

ARM_TO_BACKFILL = {'ridge_dom': 'ridge', 'esm_dom': 'esm'}
CSV = run_mod.OUT / 'dominance_sweep.csv'


def main():
    N = run_mod.RESERVED_N
    assert N == 24, N
    tasks = load_development()
    assert len(tasks) == 207
    headroom, wt_success = diag_mod.categorize(tasks)
    assert len(headroom) == 18 and len(wt_success) == 80, (len(headroom), len(wt_success))
    esm = ESMFeatures()
    csv_df = pd.read_csv(CSV).set_index(['backfill', 'N'])

    per_arm = {}
    for arm in [a for a in run_mod.ARMS if a != 'wt_new']:
        backfill = ARM_TO_BACKFILL[arm]
        set_mismatch, detail = [], []
        for label, group in [('headroom', headroom), ('wt_success', wt_success)]:
            for tid, rec in group.items():
                runner_set = set(run_mod.trajectory(rec['t'], arm, esm))
                diag_set = diag_mod.dominance_seen(rec, backfill, N, esm)
                if runner_set != diag_set:
                    set_mismatch.append(tid)
                    detail.append(dict(task_id=tid, category=label,
                                        only_runner=sorted(str(i) for i in runner_set - diag_set),
                                        only_diagnostic=sorted(str(i) for i in diag_set - runner_set)))
        runner_head = sum(1 for tid, rec in headroom.items()
                          if set(run_mod.trajectory(rec['t'], arm, esm)) & rec['true_set'])
        runner_wtlost = sum(1 for tid, rec in wt_success.items()
                            if not (set(run_mod.trajectory(rec['t'], arm, esm)) & rec['true_set']))
        csv_row = csv_df.loc[(backfill, N)]
        per_arm[arm] = dict(
            set_mismatches=len(set_mismatch),
            mismatch_task_ids=set_mismatch,
            mismatch_detail=detail,
            runner_headroom_rescued=runner_head,
            runner_wt_success_lost=runner_wtlost,
            csv_headroom_rescued=int(csv_row['headroom_rescued']),
            csv_wt_success_lost=int(csv_row['wt_success_lost']),
            runner_matches_csv=bool(runner_head == int(csv_row['headroom_rescued'])
                                    and runner_wtlost == int(csv_row['wt_success_lost'])),
            per_task_identical_to_diagnostic=len(set_mismatch) == 0)
    report = dict(
        N=N,
        tasks_207=207,
        headroom=18,
        wt_success=80,
        all_tasks_per_task_identical=bool(all(per_arm[a]['per_task_identical_to_diagnostic'] for a in per_arm)),
        all_runners_match_csv=bool(all(per_arm[a]['runner_matches_csv'] for a in per_arm)),
        per_arm=per_arm)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str), flush=True)
    ok = report['all_tasks_per_task_identical'] and report['all_runners_match_csv']
    print('\n=== VERDICT ===', 'PASS: frozen runner reproduces frozen diagnostic (N=24), per-task + aggregate' if ok
          else 'FAIL: divergence found, see detail above', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
