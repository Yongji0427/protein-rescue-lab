"""Frozen evidence audit and label-free conditional-ESM signal check."""
import os
os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
from datetime import datetime,timezone
import gzip,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import spearmanr,rankdata
from pdz2018_data import ROOT,DOMAINS,AA,RT,dump,hashes,load_development
from pdz_background_model import site
from prepare_pdz_esm import MODEL,REVISION

OUT=ROOT/'experiments/pdz2018_two_checks_v1'
UNC=ROOT/'experiments/pdz2018_uncertainty_v1'
METHODS=['wt_single','esm_wt','esm_conditional','coupling_sym','wt_coupling_rank']


def freeze():
    OUT.mkdir(exist_ok=True);assert not (OUT/'freeze.json').exists()
    p=dict(version=OUT.name,scope='Post-result developmental evidence and signal checks; no new adaptive search or supervised fit',
        primary_tasks=207,available_count_tasks=140,missing_count_tasks=67,sealed_unread=['shank3','zo1'],
        evidence='Keep original point>=.5,damage<=.2,39504-family Bonferroni conditional Poisson bounds; re-score saved trajectories; unavailable stays null; no calibration refit',
        model='facebook/esm2_t12_35M_UR50D',revision=REVISION,sequence_context='Same fullPDB construct reference contexts from esm_v2;9scan sites matched; fullassay sequence unconfirmed',
        inference='Frozen EsmForMaskedLM,eval,float32,MPS,batch32;mask target residue; all171 single backgrounds x8 other sites plus9 WT contexts per3 domains =4131 inputs; no assay label input',
        scores={'esm_wt':'logP(Rmut|WT with Rsite masked)-logP(Rwt|same)',
          'esm_conditional':'logP(Rmut|D with Rsite masked)-logP(Rwt|same)',
          'coupling_sym':'0.5*((LLR(R|D)-LLR(R|WT))+(LLR(D|R)-LLR(D|WT)))',
          'wt_coupling_rank':'negative sum of average-tie ranks of public WT single affinity and coupling_sym within unqueried pool; fixedequalweights'},
        interpretation='Evolutionary logodds, not kcal/mol, binding affinity or calibrated probability. Symmetric coupling alone does not establish absolute rescue',
        evaluation='Same first8 WT; rank remaining candidates once without new feedback; potential coverage from initial8+top24, not an adaptive search result',
        methods=METHODS,metrics=['potential coverage8+top8/16/24','conditional count-supported potential coverage','paired gains/losses','per-domain consistency',
            'within-task Spearman(score,-energy residual) over remaining candidates','AP and reciprocal first-rescue rank in initially-unsolved reachable tasks','posthoc diagnostics for previously missed16 tasks'],
        no_tuning='No fitted head, model sweep, threshold change, score weighting sweep or cohort enrichment',
        signal_gate=dict(net_potential_coverage_vs_WT_at_least=3,gained_sites_at_least=2,positive_domain_net_at_least=2,
                         no_loss_of_count_supported_coverage=True),
        association_gate='coupling_sym task-macro Spearman with favorable residual>0 overall and in at least2 domains; descriptive, not independent proof',
        selection='Among esm_conditional,coupling_sym,wt_coupling_rank maximize full207 potentialcoverage32,tie higher count-supported coverage,tie listedorder',
        next='Discuss results; do not start new search,Agnes or sealed test automatically even if signal gate passes',
        data_gap='Check primary repository catalogues for replicate/calibration/construct data. No author email sent. Discovery audit cannot prove universal absence.')
    dump(OUT/'protocol.json',p)
    prior={}
    for folder in ['pdz2018_uncertainty_v1','pdz2018_background_v1','pdz2018_rescue_ranking_v1','pdz2018_esm_v2','pdz2018_portfolio_v1','pdz2018_point_dev_v1','gb1_final_holdout_v1']:
        prior.update(hashes([f for f in (ROOT/'experiments'/folder).rglob('*') if f.is_file()]))
    dump(OUT/'preserved_hashes.json',prior)
    names=['pdz_two_checks.py','verify_pdz_two_checks.py','report_pdz_two_checks.py','pdz2018_data.py','pdz2018_uncertainty.py','pdz_background_model.py','prepare_pdz_esm.py']
    paths=[ROOT/'scripts'/n for n in names]+[OUT/'protocol.json',OUT/'preserved_hashes.json',ROOT/'data/pdz2018/source/Salinas_Ranganathan_data.mat',
        ROOT/'experiments/dataset_feasibility_v1/pdz2018_calibration_QC.csv',ROOT/'data/public_reference/residue_volumes.json']
    dump(OUT/'freeze.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes(paths),model_hashes=json.loads((ROOT/'experiments/pdz2018_esm_v2/feature_freeze.json').read_text())['model_hashes']))


def check():
    for name in ['freeze.json','preserved_hashes.json']:
        obj=json.loads((OUT/name).read_text())
        for p,h in obj.get('hashes',obj).items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    for p,h in json.loads((OUT/'freeze.json').read_text())['model_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p


def sources():
    import requests
    check();dest=OUT/'sources';dest.mkdir(exist_ok=True);ledger=[]
    urls={'dryad_dataset':'https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.gk4m1',
          'dryad_files':'https://datadryad.org/api/v2/versions/22131/files',
          'publisher':'https://api.elifesciences.org/articles/34300'}
    payloads={}
    for name,url in urls.items():
        if name=='dryad_files' and 'dryad_dataset' in payloads:
            url='https://datadryad.org'+payloads['dryad_dataset']['_links']['stash:version']['href']+'/files'
        try:
            r=requests.get(url,timeout=30);r.raise_for_status();(dest/f'{name}.json').write_bytes(r.content);payloads[name]=r.json()
            ledger.append(dict(source=url,status=r.status_code,sha256=hashlib.sha256(r.content).hexdigest(),at_utc=datetime.now(timezone.utc).isoformat()))
        except Exception as e:ledger.append(dict(source=url,error=str(e),at_utc=datetime.now(timezone.utc).isoformat()))
    # Record the actual catalogue, without downloading any new assay labels.
    dump(OUT/'source_refresh.json',dict(ledger=ledger,scope='Primary catalogue refresh; stored schema and prior reporting-form audit also reviewed',
        dryad_files=[f.get('path') for f in payloads.get('dryad_files',{}).get('_embedded',{}).get('stash:files',[])],
        dryad_total=payloads.get('dryad_files',{}).get('total'),
        publisher_attachments=[{'filename':f.get('filename'),'uri':f.get('uri')} for f in payloads.get('publisher',{}).get('additionalFiles',[])],
        prior_source_trace=str((UNC/'source_trace.json').relative_to(ROOT)),author_contact='Existing draft only; not sent',
        unresolved=['per-genotype replicate selected/unselected andWT counts','PDZ3 replicate-to-energy aggregation mapping',
                    'calibration parameter uncertainty/covariance and raw standards','complete assay-construct sequences'],
        conclusion='See refreshed primary catalogues; no claim of universal absence. No additional assay-label files downloaded.'))


def load_bounds():
    doubles=pd.read_csv(UNC/'double_bounds.csv');singles=pd.read_csv(UNC/'single_bounds.csv')
    dm={}
    for r in doubles.itertuples():dm[r.domain,r.mutation1,r.mutation2]=dm[r.domain,r.mutation2,r.mutation1]=r
    sm={(r.domain,r.mutation):r for r in singles.itertuples()}
    return dm,sm


def evidence():
    check();assert not (OUT/'evidence_summary.csv').exists()
    tasks=load_development();dm,sm=load_bounds();rows=[];coverage=[]
    for tid,t in tasks.items():
        available=bool(sm[t['domain'],t['damage']].count_available)
        combos=[r['id'] for r in t['rows'] if available and bool(dm[t['domain'],t['damage'],r['id']].combo_supported)]
        full=bool(combos and sm[t['domain'],t['damage']].damage_confirmed) if available else None
        coverage.append(dict(task_id=tid,domain=t['domain'],count_available=available,point_reachable=any(r['affinity']>=.5 for r in t['rows']),
            count_reachable=bool(combos) if available else None,full_count_reachable=full))
    for folder in ['pdz2018_rescue_ranking_v1','pdz2018_esm_v2','pdz2018_portfolio_v1']:
        with gzip.open(ROOT/'experiments'/folder/'trajectories.jsonl.gz','rt') as stream:
            for line in stream:
                tr=json.loads(line);t=tasks[tr['task_id']];ids=sum(tr['batches'],[]);assert len(ids)==len(set(ids))==min(32,len(t['rows']))
                available=bool(sm[t['domain'],t['damage']].count_available);lookup={r['id']:r for r in t['rows']}
                supported=any(bool(dm[t['domain'],t['damage'],r].combo_supported) for r in ids) if available else None
                rows.append(dict(run=folder,scheme=tr['scheme'],method=tr['method'],task_id=tr['task_id'],domain=t['domain'],
                    point_success=any(lookup[r]['affinity']>=.5 for r in ids),count_available=available,count_supported=supported,
                    full_count_supported=bool(supported and sm[t['domain'],t['damage']].damage_confirmed) if available else None))
    f=pd.DataFrame(rows);f.to_csv(OUT/'rescored_trajectories.csv',index=False);pd.DataFrame(coverage).to_csv(OUT/'task_evidence.csv',index=False)
    summary=[]
    for (run,scheme,method),a in f.groupby(['run','scheme','method'],sort=False):
        b=a[a.count_available]
        summary.append(dict(run=run,scheme=scheme,method=method,tasks=len(a),point_success=int(a.point_success.sum()),
            available_tasks=len(b),point_success_available=int(b.point_success.sum()),count_supported=int(b.count_supported.sum()),
            full_count_supported=int(b.full_count_supported.sum()),missing_count_tasks=int((~a.count_available).sum())))
    pd.DataFrame(summary).to_csv(OUT/'evidence_summary.csv',index=False)
    # Explicitly retain every miss of the latest best tree and each candidate's support.
    prior=pd.read_csv(ROOT/'experiments/pdz2018_portfolio_v1/runs.csv').query("scheme=='site' and method=='tree'")
    missed=set(prior.loc[prior.reachable & ~prior.success32,'task_id']);diagnostics=[]
    for tid in sorted(missed):
        t=tasks[tid]
        for r in t['rows']:
            if r['affinity']<.5:continue
            b=dm[t['domain'],t['damage'],r['id']]
            diagnostics.append(dict(task_id=tid,domain=t['domain'],damage=t['damage'],rescue=r['id'],point_affinity=r['affinity'],
                rescue_single_affinity=float(np.exp(r['WT_log_affinity'])),count_available=bool(b.count_available),
                counting_lower=float(b.counting_lower) if b.count_available else None,count_supported=bool(b.combo_supported) if b.count_available else None))
    pd.DataFrame(diagnostics).to_csv(OUT/'missed_rescue_evidence.csv',index=False)
    check();print('Evidence rescored',len(f),'saved trajectories',flush=True)


def contexts(context):
    for target in range(9):yield 'WT',target
    helix=context['helix']
    for i,wt in enumerate(helix):
        for a in AA:
            if a==wt:continue
            for target in range(9):
                if target!=i:yield f'{wt}{i+1}{a}',target


def infer():
    check();assert not (OUT/'inference_started.json').exists();dump(OUT/'inference_started.json',dict(at_utc=datetime.now(timezone.utc).isoformat()))
    import torch,transformers
    from transformers import AutoTokenizer,EsmForMaskedLM
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=EsmForMaskedLM.from_pretrained(MODEL,local_files_only=True).eval().to('mps')
    for p in model.parameters():p.requires_grad_(False)
    vocab=[tokenizer.convert_tokens_to_ids(a) for a in AA];assert len(set(vocab))==20 and tokenizer.unk_token_id not in vocab
    start=time.perf_counter();manifest=[]
    for context in json.loads((ROOT/'experiments/pdz2018_esm_v2/sequence_mapping.json').read_text()):
        inputs=list(contexts(context));assert len(inputs)==1377;logs=[];seq=context['sequence'];offset=context['helix_start_0based']
        for begin in range(0,len(inputs),32):
            group=inputs[begin:begin+32];sequences=[]
            for damage,target in group:
                s=list(seq)
                if damage!='WT':
                    pos=site(damage);assert s[offset+pos]==damage[0] and pos!=target;s[offset+pos]=damage[-1]
                sequences.append(''.join(s))
            encoded=tokenizer(sequences,return_tensors='pt',padding=True);positions=torch.tensor([offset+j+1 for _,j in group])
            assert encoded['input_ids'].shape[1]==len(seq)+2
            encoded['input_ids'][torch.arange(len(group)),positions]=tokenizer.mask_token_id
            assert torch.all((encoded['input_ids']==tokenizer.mask_token_id).sum(1)==1)
            encoded={k:v.to('mps') for k,v in encoded.items()}
            with torch.inference_mode():
                logits=model(**encoded).logits[torch.arange(len(group),device='mps'),positions.to('mps')]
                logs.append(logits.log_softmax(-1)[:,vocab].cpu().numpy())
        logp=np.concatenate(logs);assert logp.shape==(1377,20) and np.isfinite(logp).all()
        path=OUT/f"{context['domain']}_logprobs.npz"
        np.savez_compressed(path,background=np.array([d for d,_ in inputs]),target=np.array([j for _,j in inputs]),logp=logp)
        manifest.append(dict(domain=context['domain'],contexts=len(inputs),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print('Scored',context['domain'],len(inputs),'masked contexts',flush=True)
    check();dump(OUT/'inference_manifest.json',dict(status='completed',contexts=4131,model_forward_batches=sum((r['contexts']+31)//32 for r in manifest),
        domains=manifest,seconds=time.perf_counter()-start,torch=torch.__version__,transformers=transformers.__version__,assay_labels_loaded=False,sealed_domains_loaded=False))


class Scores:
    def __init__(self):
        self.data={};self.helix={d:seq for d,(seq,_) in DOMAINS.items()}
        for domain in DOMAINS:
            with np.load(OUT/f'{domain}_logprobs.npz') as z:
                self.data[domain]={(str(d),int(j)):row for d,j,row in zip(z['background'],z['target'],z['logp'])}
    def llr(self,domain,background,r):
        p=self.data[domain][background,site(r)]
        return float(p[AA.index(r[-1])]-p[AA.index(r[0])])
    def coupling(self,domain,d,r):
        return .5*(self.llr(domain,d,r)-self.llr(domain,'WT',r)+self.llr(domain,r,d)-self.llr(domain,'WT',d))
    def task(self,t):
        d=t['damage'];domain=t['domain'];rows=t['rows']
        return dict(wt_single=np.array([r['WT_log_affinity'] for r in rows]),
            esm_wt=np.array([self.llr(domain,'WT',r['id']) for r in rows]),
            esm_conditional=np.array([self.llr(domain,d,r['id']) for r in rows]),
            coupling_sym=np.array([self.coupling(domain,d,r['id']) for r in rows]))


def ordered(t,base,method):
    ids=np.array([r['id'] for r in t['rows']]);initial=list(map(int,np.lexsort((ids,-base['wt_single']))[:8]))
    unseen=np.array([i for i in range(len(ids)) if i not in initial],dtype=int)
    if method=='wt_coupling_rank':s=-rankdata(-base['wt_single'][unseen],method='average')-rankdata(-base['coupling_sym'][unseen],method='average')
    else:s=base[method][unseen]
    return initial,[int(unseen[i]) for i in np.lexsort((ids[unseen],-s))]


def evaluate():
    check();assert not (OUT/'signal_tasks.csv').exists() and (OUT/'inference_manifest.json').exists()
    tasks=load_development();scores=Scores();dm,sm=load_bounds();rows=[];predictions=[]
    for tid,t in tasks.items():
        base=scores.task(t);labels=np.array([r['affinity']>=.5 for r in t['rows']]);available=bool(sm[t['domain'],t['damage']].count_available)
        count=np.array([bool(dm[t['domain'],t['damage'],r['id']].combo_supported) for r in t['rows']]) if available else None
        epsilon=np.array([r['deltaG_kcal']+RT*(t['damage_log_affinity']+r['WT_log_affinity']) for r in t['rows']])
        for method in METHODS:
            initial,order=ordered(t,base,method);chosen=initial+order[:max(0,min(32,len(labels))-len(initial))]
            positives=labels[order];positions=np.flatnonzero(positives)+1
            ap=float(np.mean(np.arange(1,len(positions)+1)/positions)) if len(positions) else None
            corr=float(spearmanr(base[method][order],-epsilon[order]).statistic) if method!='wt_coupling_rank' else None
            if corr is not None and not np.isfinite(corr):corr=None
            rows.append(dict(task_id=tid,domain=t['domain'],damage=t['damage'],damage_site=site(t['damage'])+1,method=method,
                reachable=bool(labels.any()),initial_success=bool(labels[initial].any()),remaining_true=int(positives.sum()),
                **{f'potential{k}':bool(labels[(initial+order[:max(0,min(k,len(labels))-len(initial))])].any()) for k in [16,24,32]},
                count_available=available,count_supported=bool(count[chosen].any()) if available else None,
                full_count_supported=bool(count[chosen].any() and sm[t['domain'],t['damage']].damage_confirmed) if available else None,
                average_precision=ap,reciprocal_first_rescue=float(1/positions[0]) if len(positions) else None,spearman_favorable_epsilon=corr))
            for rank,i in enumerate(order,1):
                r=t['rows'][i]
                predictions.append(dict(task_id=tid,method=method,rescue=r['id'],remaining_rank=rank,point_affinity=r['affinity'],
                    epsilon_G=float(epsilon[i]),count_available=available,count_supported=bool(count[i]) if available else None))
    pd.DataFrame(rows).to_csv(OUT/'signal_tasks.csv',index=False);pd.DataFrame(predictions).to_csv(OUT/'signal_rankings.csv.gz',index=False,compression='gzip')
    check();dump(OUT/'execution.json',dict(status='completed',tasks=207,methods=len(METHODS),new_adaptive_queries=0,new_supervised_fits=0,new_agnes_calls=0,sealed_domains_loaded=False))
    print('Signal ranking checks completed',flush=True)


if __name__=='__main__':
    {'freeze':freeze,'sources':sources,'evidence':evidence,'infer':infer,'evaluate':evaluate}[sys.argv[1]]()
