"""Small budget portfolios and nested damage-weighted residual heads."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from sklearn.ensemble import HistGradientBoostingRegressor
from pdz2018_data import ROOT,DOMAINS,RT
from pdz_background_model import OnlinePredictor,site
from pdz2018_policy import PDZPolicy
from pdz_rescue_ranking import Model,public_task,features,THRESHOLD

OUT=ROOT/'experiments/pdz2018_portfolio_v1'
OLD=ROOT/'experiments/pdz2018_rescue_ranking_v1'
ESM=ROOT/'experiments/pdz2018_esm_v2'
PORTFOLIOS=['portfolio3','portfolio2','portfolio1']
HEADS=['weighted_ridge','tree','weighted_tree']
METHODS=['wt_rank','contact_seed0','ridge_full_lcb','esm_full_lcb']+PORTFOLIOS+HEADS
GRID={'weighted_ridge':[{'kind':'ridge','weight':w,'depth':0} for w in [2.,4.]],
      'tree':[{'kind':'tree','weight':1.,'depth':d} for d in [2,3]],
      'weighted_tree':[{'kind':'tree','weight':w,'depth':d} for w in [2.,4.] for d in [2,3]]}


def fit_head(records,x,config):
    y=np.array([r['epsilon'] for r in records])
    damage=np.array([max(r['gd'],r['gr'])>=-RT*np.log(.2) for r in records])
    weights=np.where(damage,config['weight'],1.);weights/=weights.mean()
    if config['kind']=='tree':
        model=HistGradientBoostingRegressor(loss='squared_error',learning_rate=.05,max_iter=150,
            max_depth=config['depth'],max_leaf_nodes=2**config['depth'],min_samples_leaf=40,
            l2_regularization=10.,early_stopping=False,random_state=20261008)
        return model.fit(x,y,sample_weight=weights)
    model=Model.__new__(Model);model.center=x.mean(0);model.scale=x.std(0);model.scale[model.scale<1e-12]=1
    z=(x-model.center)/model.scale;design=np.c_[z,np.ones(len(z))]
    penalty=np.eye(design.shape[1])*10;penalty[-1,-1]=0
    coef=cho_solve(cho_factor(design.T@(weights[:,None]*design)+penalty),design.T@(weights*y))
    model.coef,model.intercept=coef[:-1],coef[-1]
    return model


class Policy:
    def __init__(self,t,method,bases):
        assert all(set(r)=={'id','WT_log_affinity'} for r in t['rows'])
        seed=np.random.SeedSequence([20261008,list(DOMAINS).index(t['domain']),site(t['damage'])+1,ord(t['damage'][-1]),0])
        self.wt=PDZPolicy(t,'no_feedback_contact' if method=='contact_seed0' else 'wt_rank',seed)
        self.method=method
        names=['ridge','esm'] if method in PORTFOLIOS else [method if method in HEADS else ('esm' if method=='esm_full_lcb' else 'ridge')]
        self.predictors={name:OnlinePredictor(t,bases[name],True) for name in names}

    def proposal(self):
        seen=self.wt.seen;size=min(4,self.wt.cap-len(seen))
        if len(seen)<8 or self.method in ['wt_rank','contact_seed0']:return self.wt.proposal()
        if self.method not in PORTFOLIOS:return next(iter(self.predictors.values())).proposal(size,'lcb')
        remaining=[int(i) for i in self.wt.order if i not in seen]
        if size<4:return remaining[:size]
        round_=(len(seen)-8)//4
        if self.method=='portfolio3':slots=['wt','wt','wt',['ridge','esm','contact'][round_%3]]
        elif self.method=='portfolio2':slots=['wt','wt','ridge',['esm','contact'][round_%2]]
        else:slots=['wt','ridge','esm','contact']
        orders={'wt':remaining,**{k:p.proposal(len(remaining),'lcb') for k,p in self.predictors.items()}}
        contact_order=np.lexsort((self.wt.ids,-self.wt.wt,-self.wt.contact));batch=[]
        for slot in slots:
            available=[i for i in remaining if i not in batch]
            if slot=='contact':
                positive=[int(i) for i in contact_order if i in available and self.wt.contact[i]>0]
                i=positive[0] if positive else int(self.wt.rng.choice(available))
            else:i=next(int(i) for i in orders[slot] if i in available)
            batch.append(i)
        return batch

    def update(self,batch,energy):
        for p in self.predictors.values():p.update(batch,np.asarray(energy)-p.additive[batch])
        self.wt.update([str(self.wt.ids[i]) for i in batch])


def rollout(t,method,bases):
    public=public_task(t);p=Policy(public,method,bases)
    energy=np.array([r['deltaG_kcal'] for r in t['rows']]);ids=[r['id'] for r in t['rows']]
    seen=[];batches=[];first=None
    while len(seen)<p.wt.cap:
        batch=p.proposal();assert len(set(batch))==len(batch) and not set(batch)&set(seen)
        assert len(batch)==min(4,p.wt.cap-len(seen))
        # Whole batch committed before any labels enter policy state.
        p.update(batch,energy[batch]);seen+=batch;batches.append([ids[i] for i in batch])
        if first is None and min(energy[seen])<=THRESHOLD:first=len(seen)
    return dict(queries=len(seen),**{f'success{k}':bool(min(energy[seen[:k]])<=THRESHOLD) for k in [8,16,24,32]},
        first_batch_cost=first,penalized_cost=first if first else p.wt.cap+4,
        best_affinity=float(np.exp(-min(energy[seen])/RT)),reachable=bool(min(energy)<=THRESHOLD)),batches


def select_configs(records,x,tasks,scheme,held,outer_mask):
    scores={family:[dict(config=c,successes=0,tasks=0,cost=0.) for c in configs] for family,configs in GRID.items()};audit=[]
    for inner in range(3):
        excluded={i for i in range(9) if i%3==inner}
        mask=outer_mask & np.array([not excluded & {r['left_site'],r['right_site']} for r in records])
        train=[r for r,m in zip(records,mask) if m]
        selected={}
        for tid,t in tasks.items():
            ds=site(t['damage'])
            if ds not in excluded or (scheme=='site' and ds==held) or (scheme=='domain' and t['domain']==held):continue
            rows=[r for r in t['rows'] if scheme!='site' or site(r['id'])!=held]
            if rows:selected[tid]={**t,'rows':rows}
        # No inner candidate touches the outer held site or comes from the outer held domain.
        task_features={tid:features(public_task(t)) for tid,t in selected.items()}
        for family,entries in scores.items():
            for entry in entries:
                model=fit_head(train,x[mask],entry['config'])
                for tid,t in selected.items():
                    stats,_=rollout(t,family,{family:model.predict(task_features[tid])})
                    entry['successes']+=int(stats['success32']);entry['tasks']+=1;entry['cost']+=stats['penalized_cost']
        audit.append(dict(inner=inner,excluded_sites=sorted(excluded),train_genotypes=len(train),validation_tasks=len(selected)))
    selected={}
    for family,entries in scores.items():
        best=sorted(entries,key=lambda e:(-e['successes'],e['cost'],e['config']['depth'],e['config']['weight']))[0]
        selected[family]=best['config']
    return selected,dict(folds=audit,grid_scores=scores)
