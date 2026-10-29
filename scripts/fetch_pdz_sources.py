"""Retrieve public source data with provenance and CRC; no outcomes analyzed."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import struct
import sys
import zlib
import requests
ROOT=Path(__file__).resolve().parents[1]


def extract(name):
    base=ROOT/'data/pdz2026/source';entries=json.loads((base/'archive_inventory.json').read_text());f=next(f for f in entries if f['name']==name)
    out=base/'extracted'/name;out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():return name,'cached',out.stat().st_size
    url=json.loads((base/'record_metadata.json').read_text())['files'][0]['links']['self']
    def read(start,end):
        response=requests.get(url+f'?range={start}-{end}',headers={'Range':f'bytes={start}-{end}'},timeout=90)
        response.raise_for_status();assert response.status_code==206
        assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
        assert len(response.content)==end-start+1
        return response.content
    h=struct.unpack('<4s5H3I2H',read(f['offset'],f['offset']+29));assert h[0]==b'PK\x03\x04'
    start=f['offset']+30+h[-2]+h[-1];b=read(start,start+f['compressed_size']-1)
    payload=zlib.decompress(b,-15) if f['method']==8 else b
    assert len(payload)==f['size'] and zlib.crc32(payload)==f['crc']
    out.write_bytes(payload)
    out.with_suffix(out.suffix+'.provenance.json').write_text(json.dumps(dict(source_record='https://zenodo.org/records/15978782',archive_path=name,sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload),zip_crc_verified=True),indent=2)+'\n')
    return name,'downloaded',len(payload)


def main():
    if sys.argv[1]=='2018':
        out=ROOT/'data/pdz2018/source/Salinas_Ranganathan_data.mat'
        url='https://cdn.elifesciences.org/articles/34300/elife-34300-supp1-v3.mat'
        response=requests.get(url,timeout=60);response.raise_for_status();assert len(response.content)==1319663
        assert hashlib.md5(response.content).hexdigest()=='3c477dfd0d5b4e24891feae3dbaa525e'
        out.write_bytes(response.content)
        out.with_suffix('.provenance.json').write_text(json.dumps(dict(url=url,record='https://doi.org/10.5061/dryad.gk4m1',sha256=hashlib.sha256(response.content).hexdigest(),source_md5_verified=True),indent=2)+'\n')
        from scipy.io import whosmat
        print('MAT schema only:',whosmat(out))
    else:
        names = sys.argv[2:]
        if sys.argv[1]=='2026-all':
            inventory=json.loads((ROOT/'data/pdz2026/source/archive_inventory.json').read_text())
            names=[f['name'] for f in inventory if f['name'].startswith('data/inputs_mochi/') and f['name'].endswith('.txt')]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for item in pool.map(extract,names):print(item,flush=True)


if __name__=='__main__':main()
