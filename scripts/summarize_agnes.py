"""Verify official AGH traces and compare the exact same preliminary cohort."""
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import numpy as np

from agnes_client import read_config


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/gb1_agnes_v1"
PILOT = ROOT / "experiments/gb1_pilot_v1"


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    selected = protocol["damage_tasks"]
    singles = pd.read_csv(ROOT / "data/public_reference/singles.csv").set_index("mutant")
    key = read_config()["AGNES_API_KEY"]
    records, ledger, queries, decisions, session_ids, warnings = [], [], [], [], [], []
    exported = OUT / "queried_results"
    exported.mkdir(exist_ok=True)
    for damage in selected:
        trial = json.loads((OUT / "agh_events" / f"{damage}.run.json").read_text())
        state = json.loads((ROOT / ".local/agnes_evaluator" / f"{damage}.json").read_text())
        rows = state["queries"]
        assert len(rows) <= 32 and len({x["rescue"] for x in rows}) == len(rows)
        for row in rows:
            assert int(row["rescue"][1:-1]) != int(damage[1:-1])
            assert int(row["rescue"][1:-1]) not in protocol["holdout_full_sequence_positions"]
            assert np.isclose(row["WT_rescue_lnW"], singles.loc[row["rescue"], "lnW"])
            queries.append({"damage": damage, **row})
        assert int(damage[1:-1]) not in protocol["holdout_full_sequence_positions"]
        for decision in state["decisions"]:
            menu = decision["menu"]
            allowed = {x["id"] for x in menu["candidates"]}
            assert len(decision["selected"]) == 4 and len(set(decision["selected"])) == 4
            assert set(decision["selected"]) <= allowed
            assert menu["required_WT_anchor"] in decision["selected"]
            assert set(decision["selected"]).isdisjoint({x["rescue"] for x in rows[:decision["budget_before"]]})
            decisions.append({"damage": damage, **decision})
        events_text = (OUT / "agh_events" / f"{damage}.jsonl").read_text()
        assert key not in events_text
        events = list({json.loads(x)["id"]: json.loads(x) for x in events_text.splitlines()}.values())
        calls = [x for x in events if x["type"] == "tool/call"]
        assert all(x["data"]["name"] == "query_rescue_batch" for x in calls)
        for decision in state["decisions"]:
            assert any(x["data"]["args"].get("rescue_ids") == decision["selected"] for x in calls)
        for event in events:
            if event["type"] == "request/sent":
                assert event["data"]["model"]["id"] == protocol["model"]
            if event["type"] == "cost/ledger":
                assert event["data"]["model"] == protocol["model"]
                ledger.append({"damage": damage, **event["data"]})
        session_ids.append(trial["session_id"])
        if len(rows) != 32:
            warnings.append({"damage": damage, "queries": len(rows), "outcome": trial})
        shutil.copyfile(ROOT / ".local/agnes_evaluator" / f"{damage}.json", exported / f"{damage}.json")
        for budget in [8, 16, 32]:
            available = rows[:budget]
            best = max(float(singles.loc[damage, "W_display"]), max(x["W_display"] for x in available))
            first = next((x["batch_cost"] for x in available if x["W_display"] >= .5), None)
            records.append({"damage": damage, "method": "agnes_AGH", "repeat": 0, "budget": budget,
                "actual_queries": len(available), "success": best >= .5, "count_robust_success": max(x["count_ci_low"] for x in available) >= .5,
                "best_W": best, "first_success_batch_cost": first, "complete": len(rows) == 32})
    assert len(set(session_ids)) == len(selected), "Tasks must use separate AGH sessions"
    agnes = pd.DataFrame(records)
    agnes.to_csv(OUT / "runs.csv", index=False)
    pd.DataFrame(queries).to_csv(OUT / "query_log.csv", index=False)
    (OUT / "decisions.json").write_text(json.dumps(decisions, indent=2)+"\n")
    (OUT / "cost_ledger.json").write_text(json.dumps(ledger, indent=2)+"\n")
    pilot = pd.read_csv(PILOT / "runs.csv")
    pilot = pilot.loc[pilot.damage.isin(selected)]
    adaptive = pd.read_csv(ROOT / "experiments/gb1_adaptive_v1/runs.csv")
    adaptive = adaptive.loc[adaptive.damage.isin(selected) & (adaptive.repeat == 0) & adaptive.budget.isin([8,16,32])]
    all_runs = pd.concat([agnes, pilot, adaptive], ignore_index=True)
    summary = all_runs.groupby(["method", "budget"]).agg(success_rate=("success", "mean"),
        count_robust_success_rate=("count_robust_success", "mean"), mean_best_W=("best_W", "mean"), tasks=("damage", "nunique")).reset_index()
    summary.to_csv(OUT / "comparison.csv", index=False)
    cohort = pd.read_csv(PILOT / "development_tasks.csv").query("damage in @selected")
    cohort.to_csv(OUT / "cohort_audit.csv", index=False)
    total_tokens = {name: sum(x.get("tokens", {}).get(name, 0) for x in ledger) for name in ["input", "output", "cacheRead", "reasoning"]}
    audit = {"status": "passed", "tasks": len(selected), "complete_tasks": len(selected)-len(warnings),
        "reachability_tasks": int((cohort.rescue_candidates > 0).sum()), "point_queries": len(queries),
        "model_selected_batches": len(decisions), "ledger_inference_calls": len(ledger),
        "tokens": total_tokens, "holdout_labels_evaluated": 0, "credential_leak_matches": 0,
        "unique_AGH_sessions": len(set(session_ids)), "warnings": warnings,
        "protocol_sha256": hashlib.sha256((OUT / "protocol.json").read_bytes()).hexdigest(),
        "official_AGH_commit": (ROOT / "vendor/agnes-harness/.git/shallow").read_text().strip(),
        "evidence_boundary": "32 selected development tasks, one model trial/task, common first8, candidate-menu-restricted decision study; not full 188-task or final-holdout improvement."}
    (OUT / "verification.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
