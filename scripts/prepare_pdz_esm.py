"""Label-free full-construct ESM features for all designed development variants."""
import os
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
from datetime import datetime,timezone
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np
from pdz2018_data import ROOT,DOMAINS,AA,dump,hashes
from prepare_structure_prior import AA3

OUT=ROOT/'experiments/pdz2018_esm_v2'
REVISION='6fbf070e65b0b7291e7bbcd451118c216cff79d8'
MODEL=Path.home()/'.cache/huggingface/hub/models--facebook--esm2_t12_35M_UR50D/snapshots'/REVISION


def variants(helix):
    yield 'WT',[]
    singles=[(f'{native}{i+1}{a}',i,a) for i,native in enumerate(helix) for a in AA if a!=native]
    for m,i,a in singles:yield m,[(i,a)]
    for i,j in itertools.combinations(range(9),2):
        for a in AA:
            if a==helix[i]:continue
            for b in AA:
                if b!=helix[j]:yield f'{helix[i]}{i+1}{a}:{helix[j]}{j+1}{b}',[(i,a),(j,b)]


def freeze():
    OUT.mkdir(exist_ok=True);assert not (OUT/'feature_freeze.json').exists()
    mapping=[]
    prior=json.loads((ROOT/'experiments/pdz2018_point_dev_v1/structure_mapping.json').read_text())
    for domain,(helix,pdb) in DOMAINS.items():
        p=ROOT/f'data/pdz2018/public/{pdb}.pdb';lines=p.read_text().splitlines()
        seq=''.join(AA3[a] for l in lines if l.startswith('SEQRES') and l[11]=='A' for a in l[19:70].split())
        declared={int(l[13:17]) for l in lines if l.startswith('SEQRES') and l[11]=='A'}
        assert declared=={len(seq)} and seq.count(helix)==1 and set(seq)<=set(AA)
        old=next(r for r in prior if r['domain']==domain)
        observed={}
        for l in lines:
            if l.startswith('ENDMDL'):break
            if l.startswith('ATOM  ') and l[21]=='A':observed.setdefault(l[22:27].strip(),AA3[l[17:20]])
        assert ''.join(observed[k] for k in old['residue_numbers'])==helix
        mapping.append(dict(domain=domain,pdb=pdb,chain='A',sequence=seq,length=len(seq),helix=helix,
            helix_start_0based=seq.index(helix),pdb_residue_numbers=old['residue_numbers'],
            seqadv=[l.strip() for l in lines if l.startswith('SEQADV')],
            limitation='Full deposited PDB construct reference; full assay plasmid sequence not established. Scan9 residues exactly matched.'))
    dump(OUT/'sequence_mapping.json',mapping)
    protocol=dict(model='facebook/esm2_t12_35M_UR50D',revision=REVISION,layer='final12',device='mps',dtype='float32',batch=32,
        library='All WT,171 singles,12996 doubles per domain;39504 designed genotypes; no outcome filtering',
        scope='Only pdz2,pdz3,syntrophin; no sealed-domain sequences or labels',
        projection='Label-independent Gaussian 480x128,seed20261008,SD1/sqrt128; fixed before inference',
        features='384 dimensions: projected full-sequence mean double-WT; projected mean of two mutated residue states minus same WT positions; projected mean double-singleA-singleB+WT',
        learning='Frozen eval-mode ESM; no backpropagation or label input',
        note='PDB construct reference contexts, not verified full assay constructs or full-length parent proteins')
    dump(OUT/'feature_protocol.json',protocol)
    np.save(OUT/'projection.npy',(np.random.default_rng(20261008).normal(size=(480,128))/np.sqrt(128)).astype('float32'))
    paths=[ROOT/'scripts/prepare_pdz_esm.py',OUT/'sequence_mapping.json',OUT/'feature_protocol.json',OUT/'projection.npy',
           ROOT/'scripts/pdz2018_data.py',ROOT/'scripts/prepare_structure_prior.py']
    paths += [ROOT/f'data/pdz2018/public/{p}.pdb' for _,p in DOMAINS.values()]
    local_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in MODEL.iterdir() if p.is_file()}
    dump(OUT/'feature_freeze.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes(paths),model_hashes=local_hashes))


def check():
    freeze=json.loads((OUT/'feature_freeze.json').read_text())
    for name,digest in freeze['hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for name,digest in freeze['model_hashes'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name


def extract():
    check()
    import torch,transformers
    from transformers import AutoTokenizer,EsmModel
    assert torch.backends.mps.is_available()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=EsmModel.from_pretrained(MODEL,local_files_only=True,add_pooling_layer=False).eval().to('mps')
    for p in model.parameters():p.requires_grad_(False)
    projection=torch.from_numpy(np.load(OUT/'projection.npy')).to('mps')
    manifest=[];start=time.perf_counter()
    for context in json.loads((OUT/'sequence_mapping.json').read_text()):
        domain,seq,offset=context['domain'],context['sequence'],context['helix_start_0based']
        destination=OUT/f'{domain}_embeddings.npz';assert not destination.exists(),'Never overwrite extraction'
        items=list(variants(context['helix']));assert len(items)==13168
        pools=[];locals_=[]
        for begin in range(0,len(items),32):
            sequences=[]
            for key,edits in items[begin:begin+32]:
                s=list(seq)
                for i,a in edits:s[offset+i]=a
                sequences.append(''.join(s))
            encoded={k:v.to('mps') for k,v in tokenizer(sequences,padding=True,return_tensors='pt').items()}
            assert encoded['input_ids'].shape[1]==len(seq)+2
            with torch.inference_mode():
                h=model(**encoded).last_hidden_state[:,1:len(seq)+1]
                pools.append((h.mean(1)@projection).cpu().numpy())
                locals_.append((h[:,offset:offset+9]@projection).cpu().numpy())
            if begin%1024==0:print(domain,begin,'/',len(items),'seconds',round(time.perf_counter()-start,1),flush=True)
        pool=np.concatenate(pools);local=np.concatenate(locals_);keys=np.array([k for k,_ in items]);assert np.isfinite(pool).all() and np.isfinite(local).all()
        np.savez_compressed(destination,keys=keys,pool=pool,local=local)
        manifest.append(dict(domain=domain,variants=len(keys),length=len(seq),sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
        print('Saved',domain,flush=True)
    check()
    dump(OUT/'embedding_manifest.json',dict(status='completed',domains=manifest,total_variants=39504,seconds=time.perf_counter()-start,
        torch=torch.__version__,transformers=transformers.__version__,device='mps',assay_labels_loaded=False,sealed_domains_loaded=False))


if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='extract':extract()
    else:raise ValueError('freeze or extract')
