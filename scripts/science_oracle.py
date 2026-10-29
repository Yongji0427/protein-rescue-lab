"""Persistent, session-bound, idempotent budget enforcement for new experiments."""
import hashlib
import json
import math
from pathlib import Path
import secrets
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer


class ScienceTask:
    def __init__(self,path,task_id,version,rows,initial_ids,public_candidates):
        self.path=Path(path);self.rows=rows;self.public_candidates=public_candidates
        self.session_key=version+'-'+task_id
        fingerprint=hashlib.sha256(json.dumps(rows,sort_keys=True,allow_nan=False).encode()).hexdigest()
        if self.path.exists():
            self.state=json.loads(self.path.read_text())
            assert self.state['source_fingerprint']==fingerprint and self.state['session_key']==self.session_key
        else:
            assert len(initial_ids)==8 and len(set(initial_ids))==8 and set(initial_ids)<=set(rows)
            self.state=dict(task_id=task_id,task_token=secrets.token_hex(16),session_key=self.session_key,
                source_fingerprint=fingerprint,queries=[],decisions=[],requests={})
            for candidate in initial_ids:
                self.state['queries'].append(dict(query=len(self.state['queries'])+1,rescue=candidate,
                    **rows[candidate],source='shared_initial_WT'))
            self.save()
    def save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        temp=self.path.with_suffix('.tmp');temp.write_text(json.dumps(self.state,indent=2,allow_nan=False)+'\n');temp.chmod(0o600);temp.replace(self.path)
    def public(self):
        seen={r['rescue'] for r in self.state['queries']}
        return dict(task_id=self.state['task_id'],task_token=self.state['task_token'],queries_used=len(seen),
            query_budget=32,queries_remaining=32-len(seen),observed=self.state['queries'],
            candidates=[r for r in self.public_candidates if r['id'] not in seen],done=len(seen)==32)
    def handle(self,payload):
        if payload.get('session_key')!=self.session_key or payload.get('task_token')!=self.state['task_token']:
            raise ValueError('Task token is bound to one session')
        if payload.get('action')=='state':return dict(**self.public(),new_queries_charged=0)
        if payload.get('action')!='query':raise ValueError('Unknown scientific action')
        ids=payload.get('rescue_ids');pred=payload.get('predicted_residuals');rid=payload.get('request_id')
        if not isinstance(ids,list) or len(ids)!=4 or any(not isinstance(x,str) for x in ids) or len(set(ids))!=4:
            raise ValueError('Four unique IDs required')
        if any(x not in self.rows for x in ids):raise ValueError('Candidate outside query pool')
        if not isinstance(pred,list) or len(pred)!=4 or any(not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) for x in pred):
            raise ValueError('Four finite prereveal predictions required')
        if not isinstance(rid,str) or len(rid)!=64:raise ValueError('Invalid trusted request ID')
        signature=hashlib.sha256(json.dumps([ids,pred],sort_keys=True).encode()).hexdigest()
        if rid in self.state['requests']:
            if self.state['requests'][rid]!=signature:raise ValueError('Request ID cannot change payload')
            return dict(**self.public(),new_queries_charged=0,idempotent_replay=True)
        seen={r['rescue'] for r in self.state['queries']}
        # A model retry uses a new tool ID; replay only a complete previously accepted batch.
        prior=next((d for d in self.state['decisions'] if d['selected']==ids),None)
        if prior:
            return dict(**self.public(),new_queries_charged=0,idempotent_replay=True)
        if seen&set(ids):raise ValueError('Partial duplicate or already queried candidate')
        if len(seen)+4>32:raise ValueError('Query budget exhausted')
        before=len(seen)
        self.state['decisions'].append(dict(budget_before=before,selected=ids,predicted_residuals=pred,rationale=str(payload.get('rationale',''))[:300],request_id=rid))
        for candidate,prediction in zip(ids,pred):
            self.state['queries'].append(dict(query=len(self.state['queries'])+1,rescue=candidate,
                **self.rows[candidate],prereveal_prediction=prediction,source='native_scientific_tool'))
        self.state['requests'][rid]=signature;self.save()
        return dict(**self.public(),new_queries_charged=4,idempotent_replay=False)


def serve(tasks,endpoint_file):
    lock=threading.Lock();by_token={t.state['task_token']:t for t in tasks}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            try:
                if self.path!='/scientific':raise ValueError('Unknown endpoint')
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=8192:raise ValueError('Invalid request size')
                payload=json.loads(self.rfile.read(size))
                with lock:
                    token=payload.get('task_token')
                    if token not in by_token:raise ValueError('Unknown current task token')
                    value=by_token[token].handle(payload)
                status=200
            except (ValueError,TypeError,KeyError) as error:value,status=dict(error=str(error),new_queries_charged=0),400
            data=json.dumps(value,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    path=Path(endpoint_file);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(dict(url=f'http://127.0.0.1:{server.server_port}/scientific'))+'\n')
    print(json.dumps(dict(ready=True,tasks=len(tasks),port=server.server_port)),flush=True);server.serve_forever()
