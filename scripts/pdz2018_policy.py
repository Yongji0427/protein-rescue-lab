"""Transfer frozen WT/contact heuristic using only public PDZ geometry."""
import numpy as np
from pdz2018_data import ROOT


class PDZPolicy:
    def __init__(self, task, method, seed=0):
        self.ids = np.array([r['id'] for r in task['rows']])
        self.wt = np.array([r['WT_log_affinity'] for r in task['rows']])
        self.prior = task['damage_log_affinity'] + self.wt
        self.order = np.lexsort((self.ids, -self.wt))
        self.seen = []
        self.method = method
        self.rng = np.random.default_rng(seed)
        d = task['damage']
        distance = np.load(ROOT / f"data/pdz2018/public/{task['domain']}_geometry.npz")['distance']
        eligible = np.array([d[0] in 'AILMVFWY' and d[-1] in 'ACGILMFSTVWY'
            and r[0] in 'AILMVFWY' and r[-1] in 'ACGILMFSTVWY' for r in self.ids])
        self.contact = np.exp(-(distance[int(d[1:-1])-1, [int(r[1:-1])-1 for r in self.ids]]/6.)**2) * eligible
        self.cap = min(32,len(self.ids))

    def proposal(self):
        remaining = [int(i) for i in self.order if i not in self.seen]
        size = min(4, self.cap-len(self.seen))
        if self.method == 'wt_rank' or len(self.seen) < 8 or size < 4:
            return remaining[:size]
        batch = remaining[:3]
        available = [i for i in remaining if i not in batch]
        if any(self.contact[available] > 0):
            rank = np.lexsort((self.ids, -self.wt, -self.contact))
            extra = next(int(i) for i in rank if i in available)
        else:
            extra = int(self.rng.choice(available))
        return batch + [extra]

    def menu(self):
        if len(self.seen) == self.cap:
            return None
        proposal = self.proposal()
        wt = [int(i) for i in self.order if i not in self.seen][:8]
        contact = [int(i) for i in np.lexsort((self.ids,-self.wt,-self.contact))
                   if i not in self.seen and self.contact[i] > 0][:8]
        chosen = sorted(set(wt+contact+proposal),key=lambda i:self.ids[i])
        return dict(required_WT_anchor=str(self.ids[proposal[0]]),
            fixed_rule_proposal=[str(self.ids[i]) for i in proposal],
            candidates=[dict(id=str(self.ids[i]),WT_log_affinity=float(self.wt[i]),
                estimated_combo_log_affinity=float(self.prior[i]),structure_score=float(self.contact[i])) for i in chosen])

    def update(self, ids):
        indices = [int(np.flatnonzero(self.ids==r)[0]) for r in ids]
        assert len(set(indices)) == len(indices) and not set(indices)&set(self.seen)
        assert len(self.seen)+len(indices) <= self.cap
        self.seen.extend(indices)
