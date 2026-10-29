"""Build reference geometry and literature-derived volumes without assay labels."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT=Path(__file__).resolve().parents[1]
AA3=dict(zip('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split(),'ARNDCQEGHILKMFPSTWYV'))


def main():
    url='https://www.genome.jp/ftp/db/community/aaindex/aaindex1'
    response=requests.get(url,timeout=30);response.raise_for_status()
    entry=next(block.strip() for block in response.text.split('//') if block.strip().startswith('H FAUJ880103'))
    lines=entry.splitlines();start=next(i for i,line in enumerate(lines) if line.startswith('I '))
    values=[float(x) for line in lines[start+1:start+3] for x in line.split()]
    assert len(values)==20
    volumes=dict(zip('ARNDCQEGHILKMFPSTWYV',values))
    assert volumes['G']==0 and volumes['W']>volumes['V']>volumes['T']
    (ROOT/'data/provenance/FAUJ880103.txt').write_text(entry+'\n//\n')
    (ROOT/'data/public_reference/residue_volumes.json').write_text(json.dumps(volumes,indent=2)+'\n')
    provenance={'source':url,'index':'FAUJ880103','description':'Normalized van der Waals volume (Fauchere et al., 1988)',
        'record_sha256':hashlib.sha256((entry+'\n//\n').encode()).hexdigest(),'use':'ranking prior, not molecular free energy'}
    (ROOT/'data/provenance/volume_source.json').write_text(json.dumps(provenance,indent=2)+'\n')
    residues={}
    for line in (ROOT/'data/public_reference/1PGA.pdb').read_text().splitlines():
        if not line.startswith('ATOM  ') or line[21]!='A' or line[16] not in [' ','A']:continue
        assert line[26]==' '
        pos=int(line[22:26]);res=residues.setdefault(pos,{'aa':AA3[line[17:20]],'atoms':{}})
        atom=line[12:16].strip()
        res['atoms'].setdefault(atom,np.array([float(line[30:38]),float(line[38:46]),float(line[46:54])]))
    mapping=pd.read_csv(ROOT/'data/public_reference/position_mapping.csv')
    mismatches=[];coordinates=[]
    for row in mapping.itertuples():
        residue=residues[row.paper_domain_position]
        if residue['aa']!=row.reference_aa:mismatches.append({'full_position':row.full_position,'domain_position':row.paper_domain_position,'assay_aa':row.reference_aa,'structure_aa':residue['aa']})
        side=[xyz for atom,xyz in residue['atoms'].items() if atom not in ['N','CA','C','O','OXT']]
        coordinates.append(np.asarray(side or [residue['atoms']['CA']]))
    distance=np.zeros((55,55))
    for i in range(55):
        for j in range(i+1,55):
            distance[i,j]=distance[j,i]=np.linalg.norm(coordinates[i][:,None,:]-coordinates[j][None,:,:],axis=2).min()
    assert np.allclose(distance,distance.T) and np.allclose(distance.diagonal(),0)
    np.savez(ROOT/'data/public_reference/structure_geometry.npz',positions=mapping.full_position.to_numpy(),distance=distance,
        matched=np.array([row.reference_aa==residues[row.paper_domain_position]['aa'] for row in mapping.itertuples()]))
    audit={'structure':'1PGA chain A','mapped_positions':55,'matched_positions':55-len(mismatches),'mismatches':mismatches,
        'distance':'minimum WT side-chain heavy-atom distance; native glycine uses CA',
        'mismatch_policy':'zero structural score at mismatching positions, retained in full candidate pool',
        'mutation_structure_calculated':False,'assay_labels_used':False}
    (ROOT/'data/provenance/structure_mapping_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2));print('Volume index:',volumes)


if __name__=='__main__':main()
