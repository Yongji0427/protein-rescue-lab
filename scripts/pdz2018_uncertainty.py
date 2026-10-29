"""Conditional counting bounds; fixed calibration, no biological CI claim."""
import numpy as np
from scipy.stats import beta

# Fixed designed development universe, independent of observed fitness/filtering:
# three domains x (171 singles + 12996 doubles + WT), both interval tails.
DESIGNED_FAMILY = 3 * (171 + 12996 + 1)
FAMILY_ALPHA = .05
TAIL_ALPHA = FAMILY_ALPHA / (2 * DESIGNED_FAMILY)


def count_ratio_interval(selected, unselected, tail=TAIL_ALPHA):
    s,n=np.broadcast_arrays(np.asarray(selected,dtype=float),np.asarray(unselected,dtype=float))
    if np.any(~np.isfinite(s)|~np.isfinite(n)|(s<0)|(n<0)|(s!=np.floor(s))|(n!=np.floor(n))):
        raise ValueError('Counts must be finite nonnegative integers')
    lower=np.zeros(s.shape);upper=np.full(s.shape,np.inf)
    positive=s>0;denom_positive=n>0
    q=beta.ppf(tail,s[positive],n[positive]+1);lower[positive]=q/(1-q)
    q=beta.ppf(1-tail,s[denom_positive]+1,n[denom_positive]);upper[denom_positive]=q/(1-q)
    return lower,upper


def relative_interval(selected,unselected,wt_selected,wt_unselected):
    lower,upper=count_ratio_interval(selected,unselected)
    wt_lower,wt_upper=count_ratio_interval(wt_selected,wt_unselected)
    return lower/wt_upper,upper/wt_lower


def affinity_from_enrichment(relative,alpha,fraction_wt):
    raw=np.asarray(relative,dtype=float)
    fm=fraction_wt*np.power(raw,alpha)
    result=np.full(raw.shape,np.inf)
    valid=fm<1
    result[valid]=fm[valid]*(1-fraction_wt)/(fraction_wt*(1-fm[valid]))
    return result


def score_counts(selected,unselected,wt_selected,wt_unselected,alpha,fraction_wt):
    low,high=relative_interval(selected,unselected,wt_selected,wt_unselected)
    return affinity_from_enrichment(low,alpha,fraction_wt),affinity_from_enrichment(high,alpha,fraction_wt)


def missing_pdz3():
    return dict(status='unavailable',count_lower_affinity=None,count_upper_affinity=None,
        reason='Deposited raw-count arrays do not align with published PDZ3 energies; per-replicate mapping unavailable',
        biological_interval_available=False)
