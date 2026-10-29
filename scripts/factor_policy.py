"""Frozen 2x2 attribution policy: matched eligibility, feedback and geometry."""
import numpy as np
from structure_policy import StructurePolicy

METHODS = ['no_feedback_gate_random', 'no_feedback_contact', 'feedback_gate_random', 'feedback_contact']


class FactorPolicy(StructurePolicy):
    def __init__(self, ids, wt, damage_y, damage, method, seed):
        if method not in METHODS + ['wt_rank']:
            raise ValueError(method)
        self.method = method
        super().__init__(ids, wt, damage_y, damage, 'contact', seed)

    def predict(self):
        if self.method.startswith('no_feedback') or self.method == 'wt_rank':
            return self.prior.copy()
        return super().predict()

    def choose_batch(self):
        if self.method == 'wt_rank':
            seen = set(self.observed)
            return [(int(i), 'wt_rank') for i in self.order if i not in seen][:4]
        batch = super().choose_batch()
        if len(self.observed) < 8 or self.method.endswith('contact') or batch[-1][1] == 'random_full':
            return batch
        available = np.ones(len(self.ids), dtype=bool)
        available[self.observed + [i for i, _ in batch[:3]]] = False
        eligible = np.flatnonzero(available & (self.contact_score > 0))
        if not len(eligible):
            eligible = np.flatnonzero(available)
        return batch[:3] + [(int(self.rng.choice(eligible)), 'gate_random')]
