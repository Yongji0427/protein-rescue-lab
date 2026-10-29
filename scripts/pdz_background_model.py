"""Site-held-out energy residual model; only three development arrays are loaded."""
import json
import numpy as np
from scipy.io import loadmat
from scipy.linalg import cho_factor, cho_solve
from pdz2018_data import ROOT, AA, DOMAINS, RT

GROUPS = ['AVLIM', 'FWY', 'STNQ', 'KRH', 'DE', 'CGP']
VOLUME = json.loads((ROOT/'data/public_reference/residue_volumes.json').read_text())


def site(mutation):
    return int(mutation[1:-1])-1


def group(aa):
    return next(i for i, letters in enumerate(GROUPS) if aa in letters)


def onehot(index, size):
    x = np.zeros(size); x[index] = 1
    return x


def mutation_features(mutation, energy):
    native, mutant = mutation[0], mutation[-1]
    change = (VOLUME[mutant]-VOLUME[native])/8.08
    return np.r_[onehot(site(mutation),9),onehot(AA.index(native),20),
                 onehot(AA.index(mutant),20),onehot(group(mutant),6),
                 energy/RT,change,float(mutant=='P'),float(mutant=='G')]


def pair_features(d, r, gd, gr, distance, structure):
    # Canonical genotype orientation, not damage/rescue ordering.
    if site(d)>site(r): d,r,gd,gr = r,d,gr,gd
    a,b = mutation_features(d,gd), mutation_features(r,gr)
    da,db = a[-3],b[-3]
    products = np.array([gd*gr/RT**2,da*db,da+db,abs(da+db),
                         float(d[-1]=='P' or r[-1]=='P'),float(d[-1]=='G' or r[-1]=='G')])
    group_pair = np.outer(onehot(group(d[-1]),6),onehot(group(r[-1]),6)).ravel()
    x = np.r_[a,b,group_pair,products]
    if structure:
        proximity = np.exp(-(distance/6)**2)
        x = np.r_[x,distance/6,proximity,proximity*products,
                  proximity*group_pair]
    return x


def local_features(rows, damage, distance, structure):
    # Fixed local GP metric; all features are public single/sequence/WT geometry.
    x = []
    for row in rows:
        r = row['id']
        f = np.r_[onehot(site(r),9),onehot(group(r[-1]),6),
                  .5*onehot(AA.index(r[-1]),20),-row['WT_log_affinity']/4,
                  (VOLUME[r[-1]]-VOLUME[r[0]])/8.08]
        if structure: f = np.r_[f,distance[site(damage),site(r)]/6]
        x.append(f)
    return np.array(x)


def training_data():
    source = loadmat(ROOT/'data/pdz2018/source/Salinas_Ranganathan_data.mat',
                     variable_names=list(DOMAINS),squeeze_me=True,struct_as_record=False)
    assert 'shank3' not in source and 'zo1' not in source
    records = []
    for domain,(seq,_) in DOMAINS.items():
        o = source[domain]
        distance = np.load(ROOT/f'data/pdz2018/public/{domain}_geometry.npz')['distance']
        for i in range(9):
            for j in range(i+1,9):
                for a in range(20):
                    if AA[a]==seq[i] or not np.isfinite(o.deltaG_single[a,i]): continue
                    for b in range(20):
                        if AA[b]==seq[j] or not np.isfinite(o.deltaG_single[b,j]): continue
                        g,n,s = o.deltaG_double[i,j,a,b],o.unselected_double_counts[i,j,a,b],o.selected_double_counts[i,j,a,b]
                        if not np.isfinite(g) or not np.isfinite(n) or n<=0 or not np.isfinite(s): continue
                        d,r = f'{seq[i]}{i+1}{AA[a]}',f'{seq[j]}{j+1}{AA[b]}'
                        gd,gr = float(o.deltaG_single[a,i]),float(o.deltaG_single[b,j])
                        records.append(dict(domain=domain,left=d,right=r,left_site=i,right_site=j,
                            gd=gd,gr=gr,epsilon=float(g-gd-gr),distance=float(distance[i,j])))
    return records


class ResidualModel:
    def __init__(self, records, held_site, structure, alpha=10.):
        self.structure = structure
        train = [r for r in records if held_site not in (r['left_site'],r['right_site'])]
        self.train_keys = {(r['domain'],r['left'],r['right']) for r in train}
        x = np.array([pair_features(r['left'],r['right'],r['gd'],r['gr'],r['distance'],structure) for r in train])
        y = np.array([r['epsilon'] for r in train])
        self.center,self.scale = x.mean(0),x.std(0)
        self.scale[self.scale<1e-12] = 1
        z = (x-self.center)/self.scale
        self.intercept = float(y.mean())
        self.coef = cho_solve(cho_factor(z.T@z+alpha*np.eye(z.shape[1])),z.T@(y-self.intercept))
        self.audit = dict(held_site=held_site+1,structure=structure,training_genotypes=len(train),
            excluded_genotypes=len(records)-len(train),features=z.shape[1],held_site_in_training=False)

    def predict(self, task):
        dist = np.load(ROOT/f"data/pdz2018/public/{task['domain']}_geometry.npz")['distance']
        d,gd = task['damage'],-RT*task['damage_log_affinity']
        x = np.array([pair_features(d,r['id'],gd,-RT*r['WT_log_affinity'],
                     dist[site(d),site(r['id'])],self.structure) for r in task['rows']])
        for r in task['rows']:
            left,right = sorted([d,r['id']],key=site)
            assert (task['domain'],left,right) not in self.train_keys
        return self.intercept+(x-self.center)/self.scale@self.coef


class OnlinePredictor:
    def __init__(self, task, ridge_mean, structure, noise=.09):
        self.rows = task['rows']
        self.ids = np.array([r['id'] for r in self.rows])
        self.base = np.asarray(ridge_mean)
        self.additive = -RT*(task['damage_log_affinity']+np.array([r['WT_log_affinity'] for r in self.rows]))
        distance = np.load(ROOT/f"data/pdz2018/public/{task['domain']}_geometry.npz")['distance']
        f = local_features(self.rows,task['damage'],distance,structure)
        sq = np.maximum(0,(f*f).sum(1)[:,None]+(f*f).sum(1)[None,:]-2*f@f.T)
        self.kernel = np.exp(-.5*sq)
        self.noise = noise
        self.seen,self.observed = [],[]

    def update(self, indices, observed_epsilon):
        assert len(indices)==len(set(indices)) and not set(indices)&set(self.seen)
        self.seen.extend(indices);self.observed.extend(observed_epsilon)

    def posterior(self):
        if not self.seen: return self.base.copy(),self.kernel.copy()
        k = self.kernel[:,self.seen]
        factor = cho_factor(self.kernel[np.ix_(self.seen,self.seen)]+self.noise*np.eye(len(self.seen)))
        residual = np.array(self.observed)-self.base[self.seen]
        mean = self.base+k@cho_solve(factor,residual)
        cov = self.kernel-k@cho_solve(factor,k.T)
        return mean,cov

    def proposal(self, size, acquisition):
        mean,cov = self.posterior()
        remaining = [i for i in range(len(self.ids)) if i not in self.seen]
        if acquisition=='lcb':
            score = self.additive+mean-np.sqrt(np.maximum(0,np.diag(cov)))
            order = np.lexsort((self.ids,score))
            return [int(i) for i in order if i in remaining][:size]
        batch = []
        for _ in range(size):
            i = sorted(remaining,key=lambda j:(-cov[j,j],self.ids[j]))[0]
            batch.append(i);remaining.remove(i)
            v = cov[:,i].copy()
            cov -= np.outer(v,v)/(cov[i,i]+self.noise)
        return batch
