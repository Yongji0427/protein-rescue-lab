"""Same public menu builder for fixed and Agnes decisions; no labels read here."""
import numpy as np


def make_menu(policy):
    if len(policy.observed) == 32:
        return None
    available = np.ones(len(policy.ids),dtype=bool); available[policy.observed] = False
    pred = policy.predict()
    wt_top = [int(i) for i in policy.order if available[i]][:8]
    pred_top = [int(i) for i in np.lexsort((policy.ids,-pred)) if available[i]][:8]
    contact_top = [int(i) for i in np.lexsort((policy.ids,-policy.wt,-policy.contact_score))
                   if available[i] and policy.contact_score[i] > 0][:8]
    # One call per batch; public fixed proposal is cached until the next update.
    proposal = policy.choose_batch()
    chosen = set(wt_top + pred_top + contact_top + [i for i,_ in proposal])
    sources = [('WT',wt_top),('feedback_or_prior',pred_top),('structure',contact_top),('fixed_proposal',[i for i,_ in proposal])]
    return dict(required_WT_anchor=str(policy.ids[proposal[0][0]]),
                fixed_rule_proposal=[str(policy.ids[i]) for i,_ in proposal],
                candidates=[dict(id=str(policy.ids[i]),WT_lnW=round(float(policy.wt[i]),5),
                    estimated_combo_lnW=round(float(pred[i]),5),structure_score=round(float(policy.contact_score[i]),8),
                    source=[name for name,pool in sources if i in pool])
                    for i in sorted(chosen,key=lambda i:str(policy.ids[i]))])
