"""Public reference geometry changes only the exploration slot of contrast v1."""
import copy
import json
from pathlib import Path

import numpy as np
from adaptive_policy import AdaptivePolicy

ROOT = Path(__file__).resolve().parents[1]


class StructurePolicy(AdaptivePolicy):
    def __init__(self, ids, wt_lnW, damage_lnW, damage, mode, seed):
        if mode not in ['random_full', 'contact', 'packing']:
            raise ValueError(mode)
        super().__init__(ids, wt_lnW, damage_lnW, 'contrast', seed)
        self.exploration_mode = mode
        geometry = np.load(ROOT / 'data/public_reference/structure_geometry.npz')
        volumes = json.loads((ROOT / 'data/public_reference/residue_volumes.json').read_text())
        lookup = {int(p): i for i, p in enumerate(geometry['positions'])}
        di = lookup[int(damage[1:-1])]
        ri = np.array([lookup[int(r[1:-1])] for r in self.ids])
        # Scope the packing heuristic to neutral substitutions at hydrophobic native sites.
        native = 'AILMVFWY'
        neutral = 'ACGILMFSTVWY'
        eligible = np.array([damage[0] in native and damage[-1] in neutral
            and r[0] in native and r[-1] in neutral for r in self.ids])
        eligible &= geometry['matched'][di] & geometry['matched'][ri]
        proximity = np.exp(-(geometry['distance'][di, ri] / 6.0) ** 2)
        delta_d = volumes[damage[-1]] - volumes[damage[0]]
        delta_r = np.array([volumes[r[-1]] - volumes[r[0]] for r in self.ids])
        denominator = np.maximum(abs(delta_d), np.abs(delta_r))
        match = np.divide(np.minimum(abs(delta_d), np.abs(delta_r)), denominator,
                          out=np.zeros_like(delta_r), where=denominator > 0)
        match *= delta_d * delta_r < 0
        self.contact_score = proximity * eligible
        self.packing_score = self.contact_score * match

    def replicate(self, seed):
        result = copy.copy(self)
        result.rng = np.random.default_rng(seed)
        result.observed, result.labels = [], []
        return result

    def choose_batch(self):
        if len(self.observed) < 8:
            return super().choose_batch()
        available = np.ones(len(self.ids), dtype=bool)
        available[self.observed] = False
        predicted = self.predict()
        rank = np.lexsort((self.ids, -predicted))
        anchor = next(int(i) for i in self.order if available[i])
        batch = [(anchor, 'wt_anchor')]
        available[anchor] = False
        for _ in range(2):
            i = next(int(i) for i in rank if available[i])
            batch.append((i, 'adaptive'))
            available[i] = False
        score = (self.contact_score if self.exploration_mode == 'contact' else self.packing_score)
        if self.exploration_mode != 'random_full' and np.any(score[available] > 0):
            priority = np.lexsort((self.ids, -self.wt, -score))
            i = next(int(i) for i in priority if available[i])
            slot = self.exploration_mode
        else:
            i = int(self.rng.choice(np.flatnonzero(available)))
            slot = 'random_full'
        return batch + [(i, slot)]
