"""Frozen ESM augmentation of the existing small structural residual head."""
import json
import numpy as np
from scipy.special import ndtr
from pdz2018_data import ROOT,RT
from pdz_background_model import pair_features,site,OnlinePredictor
from pdz_rescue_ranking import Model,Search,THRESHOLD,public_task,features
from prepare_pdz_esm import OUT

OLD=ROOT/'experiments/pdz2018_rescue_ranking_v1'
METHODS=['wt_rank','contact_seed0','ridge_full_lcb','conservative_pi','esm_full_lcb','esm_pi']
FUSION='esm_ensemble_pi'


class ESMFeatures:
    def __init__(self):
        self.tables={};self.indices={}
        for domain in ['pdz2','pdz3','syntrophin']:
            with np.load(OUT/f'{domain}_embeddings.npz') as z:
                self.tables[domain]={k:z[k] for k in ['pool','local']}
                self.indices[domain]={k:i for i,k in enumerate(z['keys'])}

    def extra(self,domain,d,r):
        d,r=sorted([d,r],key=site);idx=self.indices[domain];values=self.tables[domain]
        a,b,c,w=idx[d],idx[r],idx[d+':'+r],idx['WT'];pool=values['pool'];local=values['local']
        i,j=site(d),site(r)
        return np.r_[pool[c]-pool[w],(local[c,i]+local[c,j]-local[w,i]-local[w,j])/2,
                     pool[c]-pool[a]-pool[b]+pool[w]].astype('float64')

    def training(self,records):
        return np.array([np.r_[pair_features(r['left'],r['right'],r['gd'],r['gr'],r['distance'],True),
                           self.extra(r['domain'],r['left'],r['right'])] for r in records])

    def task(self,t):
        return np.c_[features(t),np.array([self.extra(t['domain'],t['damage'],r['id']) for r in t['rows']])]


def search(t,method,base,calibration):
    if method in METHODS[:4]:return Search(t,method,{'ridge':base['old']},{'ridge':calibration['old']})
    if method=='esm_full_lcb':return Search(t,'ridge_full_lcb',{'ridge':base['esm']},{'ridge':calibration['esm']})
    if method=='esm_pi':return Search(t,'conservative_pi',{'ridge':base['esm']},{'ridge':calibration['esm']})
    assert method==FUSION
    return Search(t,'ensemble_pi',{'ridge':base['old'],'rank':base['esm']},
                  {'ridge':calibration['old'],'rank':calibration['esm']})


def calibrate(records,x,tasks,scheme,held,mask,store):
    values=[];audits=[]
    for inner in range(3):
        sites={s for s in range(9) if s%3==inner}
        selected=mask & np.array([not sites & {r['left_site'],r['right_site']} for r in records])
        train=[r for r,m in zip(records,selected) if m];model=Model(train,x[selected])
        n_tasks=0
        for t in tasks.values():
            ds=site(t['damage'])
            if ds not in sites or (scheme=='site' and ds==held) or (scheme=='domain' and t['domain']==held):continue
            rows=[r for r in t['rows'] if scheme!='site' or site(r['id'])!=held]
            if len(rows)<=8:continue
            n_tasks+=1;public=public_task({**t,'rows':rows})
            p=OnlinePredictor(public,model.predict(store.task(public)),True)
            order=np.lexsort((np.array([r['id'] for r in rows]),-np.array([r['WT_log_affinity'] for r in rows])))
            energy=np.array([r['deltaG_kcal'] for r in rows])
            for n in [8,16,24]:
                if len(rows)<=n:break
                batch=list(map(int,order[len(p.seen):n]));p.update(batch,energy[batch]-p.additive[batch])
                mean,cov=p.posterior();mu=p.additive+mean;sd=np.sqrt(np.maximum(np.diag(cov),1e-12)+p.noise)
                for i in order[n:]:values.append((energy[i]-mu[i],sd[i],mu[i],energy[i]<=THRESHOLD,1/(len(rows)-n)))
        audits.append(dict(inner=inner,excluded_sites=sorted(sites),training_n=len(train),calibration_tasks=n_tasks))
    e,sd,mu,y,w=np.array(values).T;w/=w.sum();bias=float(w@e);scale=max(1e-6,float(np.sqrt(w@(((e-bias)/sd)**2))))
    raw=ndtr((THRESHOLD-mu)/sd);cal=ndtr((THRESHOLD-mu-bias)/(sd*scale))
    return dict(bias=bias,scale=scale,n=len(e),raw_brier=float(w@((raw-y)**2)),calibrated_brier_fit=float(w@((cal-y)**2))),audits
