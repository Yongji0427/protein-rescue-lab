"""Evaluator-owned GB1 labels behind atomic, budgeted loopback batch queries."""
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import threading

import numpy as np
import pandas as pd

from adaptive_policy import AdaptivePolicy
from run_pilot import count_interval


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/gb1_agnes_v1"
PRIVATE = ROOT / ".local/agnes_evaluator"
PUBLIC = OUT / "public_tasks"
STATE = {}
LOCK = threading.Lock()


def save(state):
    (PRIVATE / f"{state['damage']}.json").write_text(json.dumps(state, indent=2, default=lambda x: x.item()) + "\n")


def make_menu(task):
    policy = task["policy"]
    if len(policy.observed) == 32:
        return None
    predictions = policy.predict()
    available = np.ones(len(policy.ids), dtype=bool)
    available[policy.observed] = False
    proposal = policy.choose_batch()
    chosen = {idx for idx, _ in proposal}
    # Shortlist from common public proposal generators, never hidden labels.
    wt_top = [int(i) for i in policy.order if available[i]][:8]
    pred_top = [int(i) for i in np.lexsort((policy.ids, -predictions)) if available[i]][:8]
    site_top, sites = [], set(policy.positions[policy.observed])
    for idx in policy.order:
        if available[idx] and policy.positions[idx] not in sites:
            site_top.append(int(idx)); sites.add(policy.positions[idx])
            if len(site_top) == 8:
                break
    chosen.update(wt_top + pred_top + site_top)
    task["menu_indices"] = chosen
    task["anchor"] = proposal[0][0]
    return {"required_WT_anchor": str(policy.ids[proposal[0][0]]),
        "fixed_rule_proposal": [str(policy.ids[i]) for i, _ in proposal],
        "candidates": [{"id": str(policy.ids[i]), "WT_lnW": round(float(policy.wt[i]), 5),
            "estimated_combo_lnW": round(float(predictions[i]), 5),
            "source": [label for label, pool in [("WT", wt_top), ("contrast", pred_top), ("new_site", site_top)] if i in pool]}
            for i in sorted(chosen, key=lambda i: str(policy.ids[i]))]}


def public_state(task):
    state = task["state"]
    return {"task_token": task["token"], "damage": state["damage"], "damage_lnW": state["damage_lnW"],
        "WT_lnW": 0, "damage_at_floor": state["damage_at_floor"], "query_budget": 32,
        "queries_used": len(state["queries"]), "queries_remaining": 32-len(state["queries"]),
        "rescue_threshold_W": .5, "best_W": max(q["W_display"] for q in state["queries"]),
        "observed": state["queries"], "next_batch": task["menu"],
        "done": len(state["queries"]) == 32}


def query(payload):
    token = payload.get("task_token")
    if token not in STATE:
        raise ValueError("Unknown task token")
    task = STATE[token]
    state, policy = task["state"], task["policy"]
    ids = payload.get("rescue_ids", [])
    if len(state["queries"]) >= 32:
        return public_state(task)
    if not isinstance(ids, list) or len(ids) != 4 or len(set(ids)) != 4:
        raise ValueError("Commit exactly four unique candidates")
    index = {str(policy.ids[i]): i for i in task["menu_indices"]}
    if any(i not in index for i in ids):
        raise ValueError("Candidate not in the current unqueried menu")
    indices = [index[i] for i in ids]
    if task["anchor"] not in indices:
        raise ValueError("The required WT anchor must be included")
    before = len(state["queries"])
    feedback = []
    state["decisions"].append({"budget_before": before, "menu": task["menu"], "selected": ids,
        "rationale": str(payload.get("rationale", ""))[:300]})
    for i in indices:
        row = task["labels"][i]
        record = {"query": len(state["queries"])+1, "batch_cost": before+4,
            "rescue": str(policy.ids[i]), "lnW": float(row.lnW), "W_display": float(np.exp(row.lnW)),
            "count_ci_low": float(row.ci_low), "WT_rescue_lnW": float(policy.wt[i]),
            "log_epistasis_floored": float(row.lnW-state["damage_lnW"]-policy.wt[i]), "source": "Agnes_AGH_tool"}
        state["queries"].append(record);feedback.append(row.lnW)
    policy.update(indices, feedback)
    task["menu"] = make_menu(task)
    save(state)
    return public_state(task)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        try:
            if self.path != "/query":
                raise ValueError("Only /query is available")
            size = int(self.headers.get("Content-Length", "0"))
            if size > 8192:
                raise ValueError("Request too large")
            payload = json.loads(self.rfile.read(size))
            with LOCK:
                output = query(payload)
            status = 200
        except (ValueError, TypeError, KeyError) as error:
            output, status = {"error": str(error), "new_queries_charged": 0}, 400
        data = json.dumps(output).encode()
        self.send_response(status);self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)));self.end_headers();self.wfile.write(data)


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    PRIVATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    singles = pd.read_csv(ROOT / "data/public_reference/singles.csv").set_index("mutant")
    raw = pd.read_csv(ROOT / "data/evaluator_only/gb1_raw.csv")
    held = protocol["holdout_full_sequence_positions"]
    raw = raw.loc[raw["Mut2 Position"].notna() & ~(raw["Mut1 Position"]+226).isin(held) & ~(raw["Mut2 Position"]+226).isin(held)].copy()
    raw["ci_low"], _ = count_interval(raw)
    parts = raw.mutant.str.split(":", expand=True);raw["r1"], raw["r2"] = parts[0], parts[1]
    groups1 = {k: g for k, g in raw.groupby("r1", sort=False)}
    groups2 = {k: g for k, g in raw.groupby("r2", sort=False)}
    empty = raw.iloc[:0]
    for damage in protocol["damage_tasks"]:
        frame = pd.concat([groups1.get(damage, empty), groups2.get(damage, empty)]).copy()
        frame["rescue"] = np.where(frame.r1 == damage, frame.r2, frame.r1)
        frame = frame.sort_values("rescue").reset_index(drop=True)
        seed = np.random.SeedSequence([20261008, int(damage[1:-1]), ord(damage[-1]), 0])
        policy = AdaptivePolicy(frame.rescue.to_numpy(), frame.rescue.map(singles.lnW), float(singles.loc[damage, "lnW"]), "contrast", seed)
        token = secrets.token_hex(16)
        labels = list(frame.itertuples())
        state = {"damage": damage, "damage_lnW": float(singles.loc[damage, "lnW"]),
            "damage_at_floor": bool(singles.loc[damage, "W"] <= .01), "queries": [], "decisions": []}
        for _ in range(2):
            batch = policy.choose_batch();indices, feedback = [], []
            cost = len(policy.observed)+4
            for i, _ in batch:
                row = labels[i];indices.append(i);feedback.append(row.lnW)
                state["queries"].append({"query": len(state["queries"])+1,"batch_cost": cost,
                    "rescue": str(policy.ids[i]), "lnW": float(row.lnW), "W_display": float(np.exp(row.lnW)),
                    "count_ci_low": float(row.ci_low), "WT_rescue_lnW": float(policy.wt[i]),
                    "log_epistasis_floored": float(row.lnW-state["damage_lnW"]-policy.wt[i]), "source": "shared_initial_WT"})
            policy.update(indices, feedback)
        task = {"state": state, "policy": policy, "labels": labels, "token": token}
        task["menu"] = make_menu(task);STATE[token] = task
        save(state)
        (PUBLIC / f"{damage}.json").write_text(json.dumps(public_state(task), indent=2)+"\n")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    endpoint = {"url": f"http://127.0.0.1:{server.server_port}/query"}
    Path('/tmp/protein-rescue-oracle-20261008.json').write_text(json.dumps(endpoint)+"\n")
    print(json.dumps({"ready": True, "tasks": len(STATE), "endpoint": endpoint["url"],
        "protocol_sha256": hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
