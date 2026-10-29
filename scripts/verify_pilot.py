"""Independent checks of constraints, budgets, log-derived results, and provenance."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_pilot import QueryOracle, canonical_pair


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/gb1_pilot_v1"


def must_raise(error_type, action):
    try:
        action()
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}")


def main():
    checks = []
    oracle = QueryOracle("Q228A", {"Y229C": 0.6, "K230D": 0.7}, 1)
    assert oracle.query("Y229C") == .6 and oracle.used == 1
    assert oracle.query("Y229C") == .6 and oracle.used == 1
    must_raise(RuntimeError, lambda: oracle.query("K230D"))
    must_raise(ValueError, lambda: oracle.query("Q228C"))
    must_raise(ValueError, lambda: oracle.query("Y229C:K230D"))
    assert oracle.used == 1
    assert canonical_pair("K230D", "Y229C") == canonical_pair("Y229C", "K230D")
    checks.append("cache, exhausted budget, invalid/same-site and multi-edit candidates, canonical identity")

    protocol = json.loads((OUT / "protocol.json").read_text())
    audit = json.loads((OUT / "audit.json").read_text())
    assert hashlib.sha256((OUT / "protocol.json").read_bytes()).hexdigest() == audit["protocol_sha256"]
    source = json.loads((ROOT / "data/provenance/sources.json").read_text())
    for record in source["downloads"]:
        digest = hashlib.sha256()
        with (ROOT / record["output"]).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == record["output_sha256"]
    checks.append("frozen protocol and extracted official data checksums")

    log = pd.read_csv(OUT / "query_log.csv.gz")
    keys = ["damage", "method", "repeat"]
    sizes = log.groupby(keys).size()
    assert (sizes == 32).all()
    assert not log.duplicated(keys + ["rescue"]).any()
    assert not log.duplicated(keys + ["combination"]).any()
    held = set(protocol["holdout_full_sequence_positions"])
    dpos = log.damage.str[1:-1].astype(int)
    rpos = log.rescue.str[1:-1].astype(int)
    assert (dpos != rpos).all()
    assert not dpos.isin(held).any() and not rpos.isin(held).any()
    assert (log.batch_cost == ((log["query"] + 3) // 4) * 4).all()
    expected_ids = np.where(dpos < rpos, log.damage + ":" + log.rescue, log.rescue + ":" + log.damage)
    assert (log.combination == expected_ids).all()
    runs = pd.read_csv(OUT / "runs.csv")
    tasks = pd.read_csv(OUT / "development_tasks.csv")
    assert len(sizes) == len(tasks) * 101
    for budget in protocol["budgets"]:
        subset = log.loc[log["query"] <= budget]
        actual = subset.groupby(keys).W_display.max().rename("logged_best")
        compared = runs.loc[runs.budget == budget].set_index(keys).join(actual)
        damage_w = compared.reset_index().damage.map(tasks.set_index("damage").damage_W_display).to_numpy()
        assert np.allclose(compared.best_W, np.maximum(compared.logged_best, damage_w))
        assert (compared.success == (compared.best_W >= .5)).all()
        assert (compared.queries_used == budget).all()
    checks.append(f"all {len(log):,} logged queries: unique labels, damage retained, one added edit, no holdout, exact budgets and outcomes")

    summary = pd.read_csv(OUT / "summary.csv")
    for budget in protocol["budgets"]:
        expected = tasks[f"random_exact_success_{budget}"].mean()
        observed = summary.query("method=='random' and budget==@budget").success_rate.iloc[0]
        probs = tasks[f"random_exact_success_{budget}"].to_numpy()
        se = np.sqrt(np.sum(probs * (1-probs)) / (len(tasks)**2 * 100))
        assert abs(expected - observed) < 5 * se + 1e-4
    assert (summary.success_rate <= audit["reachable_development_tasks"] / len(tasks) + 1e-12).all()
    checks.append("Monte Carlo random baseline agrees with exact hypergeometric expectation; no method exceeds reachability ceiling")
    result = {"status": "passed", "checks": checks, "log_rows": len(log), "runs": len(sizes)}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
