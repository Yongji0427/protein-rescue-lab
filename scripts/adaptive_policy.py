"""Frozen, label-free batch policy; observed combinations enter only as feedback."""
import numpy as np


AA_GROUP = {aa: group for group, letters in enumerate(
    ["AVLIM", "FWY", "STNQ", "KRH", "DE", "CGP"]
) for aa in letters}


class AdaptivePolicy:
    def __init__(self, ids, wt_lnW, damage_lnW, mode, seed):
        if mode not in ["contrast", "direct"]:
            raise ValueError(mode)
        self.ids = np.asarray(ids)
        self.wt = np.asarray(wt_lnW, dtype=float)
        self.prior = damage_lnW + self.wt
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.positions = np.array([int(value[1:-1]) for value in ids])
        groups = np.array([AA_GROUP[value[-1]] for value in ids])
        self.kernel = (
            (self.positions[:, None] == self.positions[None, :]).astype(float)
            + .25 * (groups[:, None] == groups[None, :]) + .05
        )
        self.order = np.lexsort((self.ids, -self.wt))
        self.observed = []
        self.labels = []

    def predict(self):
        if not self.observed:
            return self.prior.copy()
        weights = self.kernel[:, self.observed]
        denom = 1.0 + weights.sum(axis=1)
        observed_y = np.asarray(self.labels)
        if self.mode == "contrast":
            residual = observed_y - self.prior[self.observed]
            return self.prior + np.sum(weights * residual, axis=1) / denom
        return (self.prior + np.sum(weights * observed_y, axis=1)) / denom

    def choose_batch(self):
        available = np.ones(len(self.ids), dtype=bool)
        available[self.observed] = False
        if len(self.observed) < 8:
            chosen = [int(i) for i in self.order if available[i]][:4]
            return [(i, "initial_wt") for i in chosen]
        predicted = self.predict()
        batch = []
        anchor = next(int(i) for i in self.order if available[i])
        batch.append((anchor, "wt_anchor"))
        available[anchor] = False
        for _ in range(2):
            rank = np.lexsort((self.ids, -predicted))
            selected = next(int(i) for i in rank if available[i])
            batch.append((selected, "adaptive"))
            available[selected] = False
        unseen_sites = ~np.isin(self.positions, self.positions[self.observed + [i for i, _ in batch]])
        exploration = np.flatnonzero(available & unseen_sites)
        if not len(exploration):
            exploration = np.flatnonzero(available)
        batch.append((int(self.rng.choice(exploration)), "exploration"))
        return batch

    def update(self, indices, labels):
        if len(indices) != 4 or len(labels) != 4:
            raise ValueError("Feedback must cover exactly one committed batch")
        if len(set(indices)) != 4 or set(indices).intersection(self.observed):
            raise ValueError("A label cannot be counted twice")
        self.observed.extend(indices)
        self.labels.extend(float(y) for y in labels)
