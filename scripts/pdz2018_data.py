"""Development PDZ data only; never load sealed SHANK3 or ZO1 arrays."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/pdz2018_point_dev_v1'
AA = 'ARNDCQEGHILKMFPSTWYV'
RT = .00198720425864083 * 298.15
DOMAINS = {'pdz2': ('HEDAVAALK', '1QLC'), 'pdz3': ('HEQAAIALK', '1BE9'),
           'syntrophin': ('HDEAVQALK', '1Z86')}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_development():
    source = loadmat(ROOT / 'data/pdz2018/source/Salinas_Ranganathan_data.mat',
                     variable_names=list(DOMAINS), squeeze_me=True, struct_as_record=False)
    assert 'shank3' not in source and 'zo1' not in source
    calibration = pd.read_csv(ROOT / 'experiments/dataset_feasibility_v1/pdz2018_calibration_QC.csv').set_index('domain')
    tasks = {}
    for domain, (sequence, _) in DOMAINS.items():
        o = source[domain]
        singles = {}
        for i in range(9):
            assert AA[np.flatnonzero(~np.isfinite(o.unselected_single_counts[:, i]))[0]] == sequence[i]
            for a in range(20):
                g = o.deltaG_single[a, i]
                if AA[a] != sequence[i] and np.isfinite(g):
                    singles[f'{sequence[i]}{i+1}{AA[a]}'] = -float(g) / RT
        support = bool(calibration.loc[domain, 'count_support_available'])
        frames = {d: [] for d, y in singles.items() if y <= np.log(.2)}
        for i in range(9):
            for j in range(i+1, 9):
                for a in range(20):
                    for b in range(20):
                        d, r = f'{sequence[i]}{i+1}{AA[a]}', f'{sequence[j]}{j+1}{AA[b]}'
                        g, n, s = o.deltaG_double[i,j,a,b], o.unselected_double_counts[i,j,a,b], o.selected_double_counts[i,j,a,b]
                        if d not in singles or r not in singles or not np.isfinite(g) or not np.isfinite(n) or n <= 0 or not np.isfinite(s):
                            continue
                        low_affinity = None
                        if support:
                            from scipy.stats import beta
                            q = float(beta.ppf(.05, s, n+1)) if s > 0 else 0.
                            raw = q / (1-q) / (o.selected_wt_counts / o.unselected_wt_counts)
                            alpha, f = calibration.loc[domain, ['alpha', 'fraction_WT']]
                            fm = f * raw ** alpha
                            low_affinity = float(fm*(1-f)/(f*(1-fm))) if fm < 1 else None
                        for damage, rescue in [(d,r), (r,d)]:
                            if damage in frames:
                                frames[damage].append(dict(id=rescue, WT_log_affinity=singles[rescue],
                                    deltaG_kcal=float(g), log_affinity=-float(g)/RT,
                                    affinity=float(np.exp(-g/RT)), count_lower_affinity=low_affinity))
        for damage, rows in frames.items():
            task_id = domain + '__' + damage
            tasks[task_id] = dict(task_id=task_id, domain=domain, damage=damage,
                damage_log_affinity=singles[damage], rows=sorted(rows,key=lambda r:r['id']),
                count_support_available=support)
    return tasks


def hashes(paths):
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
