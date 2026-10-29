"""Small frozen rescue-ranking experiment; policies receive public fields only."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit, ndtr
from pdz2018_data import ROOT, RT
from pdz_background_model import pair_features, OnlinePredictor, site

METHODS = ['wt_rank', 'contact_seed0', 'ridge_full_lcb', 'conservative_lcb',
           'conservative_pi', 'rank_pi', 'ensemble_pi']
THRESHOLD = -RT*np.log(.5)


def public_task(task):
    return {**{k: task[k] for k in ['task_id','domain','damage','damage_log_affinity']},
            'rows': [{k:r[k] for k in ['id','WT_log_affinity']} for r in task['rows']]}


def features(task):
    distance = np.load(ROOT/f"data/pdz2018/public/{task['domain']}_geometry.npz")['distance']
    return np.array([pair_features(task['damage'],r['id'],-RT*task['damage_log_affinity'],
        -RT*r['WT_log_affinity'],distance[site(task['damage']),site(r['id'])],True) for r in task['rows']])


class Model:
    def __init__(self, records, x, ranking=False):
        y = np.array([r['epsilon'] for r in records])
        self.center,self.scale = x.mean(0),x.std(0)
        self.scale[self.scale<1e-12] = 1
        z = (x-self.center)/self.scale
        self.intercept = y.mean()
        self.coef = cho_solve(cho_factor(z.T@z+10*np.eye(z.shape[1])),z.T@(y-self.intercept))
        self.audit = dict(n=len(records),ranking=ranking,optimizer_success=True)
        if not ranking: return
        # Same damage background; canonical genotype rows are shared, never duplicated across folds.
        groups = {}
        additive = np.array([r['gd']+r['gr'] for r in records])
        energy = y+additive
        for i,r in enumerate(records):
            for endpoint,g in [('left','gd'),('right','gr')]:
                if r[g]>=-RT*np.log(.2):
                    groups.setdefault((r['domain'],r[endpoint]),[]).append(i)
        rng = np.random.default_rng(20261008)
        pairs = []
        for key,idx in sorted(groups.items()):
            if len(idx)<2: continue
            for _ in range(32):
                a,b = rng.choice(idx,2,replace=False)
                if abs(energy[a]-energy[b])>1e-8: pairs.append((a,b))
        assert pairs
        a,b = np.array(pairs).T
        dz = (z[a]-z[b])/RT
        da = (additive[a]-additive[b])/RT
        signs = np.sign(energy[a]-energy[b])
        reg = 10/len(y)
        # Half MSE in kcal/mol, half pairwise logistic ranking; fixed weights, no sweep.
        def objective(w):
            error = z@w[:-1]+w[-1]-y
            margin = signs*(dz@w[:-1]+da)
            loss = .5*np.mean(error**2)+.5*np.mean(np.logaddexp(0,-margin))+reg*np.sum(w[:-1]**2)
            grad = z.T@error/len(y)-.5*dz.T@(signs*expit(-margin))/len(a)+2*reg*w[:-1]
            return loss,np.r_[grad,error.mean()]
        fit = minimize(objective,np.r_[self.coef,self.intercept],jac=True,method='L-BFGS-B',
                       options={'maxiter':1000,'ftol':1e-10,'gtol':1e-6,'maxls':40})
        self.coef,self.intercept = fit.x[:-1],fit.x[-1]
        self.audit.update(optimizer_success=bool(fit.success),iterations=int(fit.nit),
                          message=str(fit.message),pairs=len(pairs),objective=float(fit.fun))
        assert fit.success,self.audit

    def predict(self,x):
        return self.intercept+(x-self.center)/self.scale@self.coef

    def save(self,path):
        np.savez(path,center=self.center,scale=self.scale,coef=self.coef,intercept=self.intercept)

    @classmethod
    def load(cls,path):
        obj=cls.__new__(cls)
        with np.load(path) as data:
            for k in ['center','scale','coef','intercept']: setattr(obj,k,data[k])
        return obj


def probability(predictor,calibration):
    residual,cov = predictor.posterior()
    mu = predictor.additive+residual
    sd = np.sqrt(np.maximum(np.diag(cov),1e-12)+predictor.noise)
    return ndtr((THRESHOLD-mu-calibration['bias'])/(sd*calibration['scale']))


class Search:
    def __init__(self,task,method,bases,calibration):
        from pdz2018_policy import PDZPolicy
        from pdz2018_data import DOMAINS
        assert all(set(r)=={'id','WT_log_affinity'} for r in task['rows'])
        seed=np.random.SeedSequence([20261008,list(DOMAINS).index(task['domain']),site(task['damage'])+1,ord(task['damage'][-1]),0])
        self.wt=PDZPolicy(task,'no_feedback_contact' if method=='contact_seed0' else 'wt_rank',seed)
        self.models={k:OnlinePredictor(task,v,True) for k,v in bases.items()}
        self.method,self.calibration=method,calibration

    def proposal(self):
        seen=self.wt.seen
        size=min(4,self.wt.cap-len(seen))
        if len(seen)<8 or self.method in ['wt_rank','contact_seed0']: return self.wt.proposal()
        if self.method=='ridge_full_lcb': return self.models['ridge'].proposal(size,'lcb')
        remaining=[int(i) for i in self.wt.order if i not in seen]
        # Three WT anchors, one expert proposal. Partial final batch remains WT.
        if size<4:return remaining[:size]
        batch=remaining[:3]
        if self.method=='conservative_lcb':
            order=self.models['ridge'].proposal(len(remaining),'lcb')
        else:
            kinds=['ridge','rank'] if self.method=='ensemble_pi' else ['rank' if self.method=='rank_pi' else 'ridge']
            p=np.mean([probability(self.models[k],self.calibration[k]) for k in kinds],axis=0)
            order=np.lexsort((self.wt.ids,-p))
        batch.append(next(int(i) for i in order if i not in seen and i not in batch))
        return batch

    def update(self,batch,energies):
        for model in self.models.values(): model.update(batch,np.asarray(energies)-model.additive[batch])
        self.wt.update([str(self.wt.ids[i]) for i in batch])
