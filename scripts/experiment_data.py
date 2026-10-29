"""Evaluator-only loading; final subset is forbidden until freeze manifest exists."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_pilot import count_interval

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'experiments/gb1_pilot_v1'


def load_frames(protocol, out):
    held = protocol['holdout_full_sequence_positions']
    if protocol['split'] == 'holdout':
        freeze = json.loads((out / 'freeze.json').read_text())
        for path, expected in freeze['hashes'].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    singles = pd.read_csv(ROOT / 'data/public_reference/singles.csv').set_index('mutant')
    # Scores are consumed only after filtering the designated split.
    raw = pd.read_csv(ROOT / 'data/evaluator_only/gb1_raw.csv')
    p1, p2 = raw['Mut1 Position'] + 226, raw['Mut2 Position'] + 226
    if protocol['split'] == 'development':
        mask = p2.notna() & ~p1.isin(held) & ~p2.isin(held)
    else:
        mask = p2.notna() & (p1.isin(held) ^ p2.isin(held))
    raw = raw.loc[mask].copy()
    raw['ci_low'], _ = count_interval(raw)
    raw['W_display'] = np.exp(raw.lnW)
    parts = raw.mutant.str.split(':', expand=True)
    raw['r1'], raw['r2'] = parts[0], parts[1]
    g1 = {k: g for k, g in raw.groupby('r1', sort=False)}
    g2 = {k: g for k, g in raw.groupby('r2', sort=False)}
    empty = raw.iloc[:0]
    frames = {}
    for damage in protocol['damage_tasks']:
        frame = pd.concat([g1.get(damage, empty), g2.get(damage, empty)]).copy()
        frame['rescue'] = np.where(frame.r1 == damage, frame.r2, frame.r1)
        frame = frame.sort_values('rescue').reset_index(drop=True)
        assert len(frame) >= 500 and not frame.rescue.duplicated().any()
        assert not frame.rescue.str[1:-1].astype(int).isin(held).any()
        assert int(damage[1:-1]) in held if protocol['split'] == 'holdout' else int(damage[1:-1]) not in held
        frames[damage] = frame
    return singles, frames
