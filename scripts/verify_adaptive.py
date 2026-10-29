"""Validate policy arithmetic, causal replay, matched starts, and all query budgets."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_policy import AdaptivePolicy


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "gb1_adaptive_v1"
PILOT = ROOT / "experiments" / "gb1_pilot_v1"


def main():
    checks = []
    ids = ["A229F", "A229Y", "D230F", "D230K", "V231A", "V231Y", "K232D", "K232P", "G233W", "G233S", "E234Q", "E234V"]
    wt = np.linspace(-1, 1, len(ids))
    observed = [0, 1, 2, 3]
    feedback = [-.1, -1, .2, -.5]
    for mode in ["contrast", "direct"]:
        policy = AdaptivePolicy(ids, wt, -2, mode, 7)
        policy.update(observed, feedback)
        k = policy.kernel[:, observed]
        if mode == "contrast":
            expected = policy.prior + (k @ (np.array(feedback) - policy.prior[observed])) / (1 + k.sum(1))
        else:
            expected = (policy.prior + k @ feedback) / (1 + k.sum(1))
        assert np.allclose(policy.predict(), expected)
        try:
            policy.update(observed, feedback)
            raise AssertionError("Duplicate feedback accepted")
        except ValueError:
            pass
    checks.append("independent arithmetic checks for residual and absolute-feedback predictions; duplicate feedback rejected")
    protocol = json.loads((OUT / "protocol.json").read_text())
    audit = json.loads((OUT / "audit.json").read_text())
    assert hashlib.sha256((OUT / "protocol.json").read_bytes()).hexdigest() == audit["protocol_sha256"]
    assert hashlib.sha256((ROOT / "scripts/adaptive_policy.py").read_bytes()).hexdigest() == audit["policy_sha256"]
    log = pd.read_csv(OUT / "query_log.csv.gz")
    keys = ["damage", "method", "repeat"]
    assert len(log) == audit["queries"]
    assert (log.groupby(keys).size() == 32).all()
    assert not log.duplicated(keys + ["rescue"]).any()
    dpos = log.damage.str[1:-1].astype(int)
    rpos = log.rescue.str[1:-1].astype(int)
    assert (dpos != rpos).all()
    assert not dpos.isin(protocol["holdout_full_sequence_positions"]).any()
    assert not rpos.isin(protocol["holdout_full_sequence_positions"]).any()
    assert (log.batch_cost == ((log["query"] + 3) // 4) * 4).all()
    starts = pd.read_csv(PILOT / "query_log.csv.gz", usecols=["damage", "method", "query", "rescue"])
    starts = starts.query("method=='wt_single_rank' and query<=8").set_index(["damage", "query"]).rescue
    first = log.loc[log["query"] <= 8].copy()
    expected = pd.MultiIndex.from_frame(first[["damage", "query"]]).map(starts)
    assert (first.rescue.to_numpy() == expected.to_numpy()).all()
    checks.append(f"all {len(log):,} queries: unique labels, fixed damage, one added edit, no reserved position, exact batch costs, identical first eight WT queries")

    # Identity-only catalog construction cannot inspect or infer unqueried scores.
    catalog = pd.read_csv(ROOT / "data/evaluator_only/gb1_raw.csv", usecols=["mutant", "Mut1 Position", "Mut2 Position"])
    held = protocol["holdout_full_sequence_positions"]
    catalog = catalog.loc[catalog["Mut2 Position"].notna() & ~(catalog["Mut1 Position"] + 226).isin(held) & ~(catalog["Mut2 Position"] + 226).isin(held)]
    parts = catalog.mutant.str.split(":", expand=True)
    catalog["r1"], catalog["r2"] = parts[0], parts[1]
    g1 = {k: g for k, g in catalog.groupby("r1", sort=False)}
    g2 = {k: g for k, g in catalog.groupby("r2", sort=False)}
    empty = catalog.iloc[0:0]
    singles = pd.read_csv(ROOT / "data/public_reference/singles.csv").set_index("mutant")
    replay_batches = 0
    for (damage, method), rows in log.loc[log.repeat == 0].groupby(["damage", "method"]):
        pool = pd.concat([g1.get(damage, empty), g2.get(damage, empty)])
        ids = np.sort(np.where(pool.r1 == damage, pool.r2, pool.r1))
        wt = singles.loc[ids, "lnW"].to_numpy()
        seed = np.random.SeedSequence([protocol["seed"], int(damage[1:-1]), ord(damage[-1]), 0])
        policy = AdaptivePolicy(ids, wt, float(singles.loc[damage, "lnW"]), method, seed)
        for cost, batch in rows.groupby("batch_cost", sort=True):
            batch = batch.sort_values("query")
            prediction = policy.predict()
            choices = policy.choose_batch()
            indices = [i for i, _ in choices]
            assert list(ids[indices]) == batch.rescue.tolist()
            assert [slot for _, slot in choices] == batch.slot.tolist()
            assert np.allclose(prediction[indices], batch.predicted_lnW_before_batch)
            # Current labels become visible only after all current choices are verified.
            policy.update(indices, batch.lnW.tolist())
            replay_batches += 1
    checks.append(f"causal batch-by-batch replay of both methods for every task at seed 0: {replay_batches:,} batches reconstructed solely from previous feedback")
    runs = pd.read_csv(OUT / "runs.csv")
    for budget in [4, 8, 12, 16, 20, 24, 28, 32]:
        observed = log.loc[log["query"] <= budget].groupby(keys).W_display.max()
        compared = runs.loc[runs.budget == budget].set_index(keys)
        initial = pd.Series(compared.index.get_level_values("damage")).map(singles.W_display).to_numpy()
        expected_best = np.maximum(observed.reindex(compared.index).to_numpy(), initial)
        assert np.allclose(compared.best_W, expected_best)
        assert (compared.success.to_numpy() == (expected_best >= .5)).all()
        lower = log.loc[log["query"] <= budget].groupby(keys).count_ci_low.max().reindex(compared.index)
        assert (compared.count_robust_success.to_numpy() == (lower.to_numpy() >= .5)).all()
    checks.append("all run-level point and count-interval success metrics reconstructed from query logs")
    result = {"status": "passed", "checks": checks, "queries": len(log), "replay_batches": replay_batches}
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
