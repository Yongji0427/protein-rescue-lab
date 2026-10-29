"""Versioned official-AGH evaluator; same menu function as fixed comparator."""
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import sys
import threading
import numpy as np
from experiment_data import ROOT, load_frames
from factor_policy import FactorPolicy
from menu_policy import make_menu

OUT=ROOT/sys.argv[1]
PRIVATE=ROOT/'.local'/OUT.name/'evaluator'
STATE={};LOCK=threading.Lock()


def save(task):
    (PRIVATE/(task['state']['damage']+'.json')).write_text(json.dumps(task['state'],indent=2)+'\n')
    (OUT/'queried_results'/(task['state']['damage']+'.json')).write_text(json.dumps(task['state'],indent=2)+'\n')


def public_state(task):
    s=task['state'];rows=s['queries']
    return dict(task_token=task['token'],damage=s['damage'],damage_lnW=s['damage_lnW'],damage_at_floor=s['damage_at_floor'],WT_lnW=0,
                query_budget=32,queries_used=len(rows),queries_remaining=32-len(rows),rescue_threshold_W=.5,
                observed=rows,best_W=max(r['W_display'] for r in rows),next_batch=task['menu'],done=len(rows)==32)


def commit(task,ids,rationale,source):
    s=task['state'];policy=task['policy'];menu=task['menu'];before=len(s['queries'])
    if before==32:return public_state(task)
    if not isinstance(ids,list) or len(ids)!=4 or len(set(ids))!=4:raise ValueError('Exactly four unique candidates required')
    allowed={c['id'] for c in menu['candidates']}
    if not set(ids)<=allowed or menu['required_WT_anchor'] not in ids:raise ValueError('Current menu and required anchor must be respected')
    index={str(r):i for i,r in enumerate(policy.ids)};indices=[index[r] for r in ids]
    if set(indices)&set(policy.observed):raise ValueError('Already queried candidate')
    s['decisions'].append(dict(budget_before=before,menu=menu,selected=ids,rationale=str(rationale)[:300],source=source))
    feedback=[]
    for i in indices:
        row=task['labels'][i];wt=float(policy.wt[i]);y=float(row.lnW)
        s['queries'].append(dict(query=len(s['queries'])+1,batch_cost=before+4,rescue=str(policy.ids[i]),lnW=y,W_display=float(np.exp(y)),
                                count_ci_low=float(row.ci_low),WT_rescue_lnW=wt,log_epistasis_floored=y-s['damage_lnW']-wt,source=source))
        feedback.append(y)
    policy.update(indices,feedback);task['menu']=make_menu(policy);save(task)
    return public_state(task)


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
        try:
            if self.path!='/query':raise ValueError('Only /query available')
            size=int(self.headers.get('Content-Length','0'))
            if size>8192:raise ValueError('Request too large')
            payload=json.loads(self.rfile.read(size))
            with LOCK:
                token=payload.get('task_token')
                if token not in STATE:raise ValueError('Unknown current task token')
                value=commit(STATE[token],payload.get('rescue_ids'),payload.get('rationale',''),'Agnes_AGH_tool')
            status=200
        except (ValueError,TypeError,KeyError) as error:value,status={'error':str(error),'new_queries_charged':0},400
        data=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)


def make_task(damage,frame,singles,method,seed):
    ids=frame.rescue.to_numpy();dy=float(singles.loc[damage,'lnW'])
    policy=FactorPolicy(ids,singles.loc[ids,'lnW'].to_numpy(),dy,damage,method,seed)
    state=dict(damage=damage,damage_lnW=dy,damage_at_floor=bool(singles.loc[damage,'W']<=.01),queries=[],decisions=[])
    labels=list(frame.itertuples())
    for _ in range(2):
        batch=policy.choose_batch();indices=[i for i,_ in batch];feedback=[];cost=len(policy.observed)+4
        for i in indices:
            row=labels[i];y=float(row.lnW);wt=float(policy.wt[i]);feedback.append(y)
            state['queries'].append(dict(query=len(state['queries'])+1,batch_cost=cost,rescue=str(ids[i]),lnW=y,W_display=float(np.exp(y)),count_ci_low=float(row.ci_low),WT_rescue_lnW=wt,log_epistasis_floored=y-dy-wt,source='shared_initial_WT'))
        policy.update(indices,feedback)
    task=dict(state=state,policy=policy,labels=labels,token=secrets.token_hex(16))
    task['menu']=make_menu(policy);return task


def main():
    p=json.loads((OUT/'protocol.json').read_text())
    assert not PRIVATE.exists(), 'Existing evaluator state must never be reset'
    PRIVATE.mkdir(parents=True,mode=0o700)
    for folder in ['public_tasks','queried_results','fixed_menu_results']:(OUT/folder).mkdir(exist_ok=True)
    singles,frames=load_frames(p,OUT)
    for damage,frame in frames.items():
        seed=np.random.SeedSequence([p['seed'],int(damage[1:-1]),ord(damage[-1]),0])
        fixed=make_task(damage,frame,singles,p['selected_method'],seed)
        while fixed['menu'] is not None:commit(fixed,fixed['menu']['fixed_rule_proposal'],'frozen fixed choice','fixed_menu')
        (OUT/'fixed_menu_results'/(damage+'.json')).write_text(json.dumps(fixed['state'],indent=2)+'\n')
        task=make_task(damage,frame,singles,p['selected_method'],seed);STATE[task['token']]=task;save(task)
        (OUT/'public_tasks'/(damage+'.json')).write_text(json.dumps(public_state(task),indent=2)+'\n')
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    endpoint=dict(url=f'http://127.0.0.1:{server.server_port}/query')
    Path('/tmp/protein-rescue-oracle-20261008.json').write_text(json.dumps(endpoint)+'\n')
    (OUT/'endpoint.json').write_text(json.dumps(endpoint)+'\n')
    print(json.dumps(dict(ready=True,tasks=len(STATE),endpoint=endpoint['url'])),flush=True);server.serve_forever()


if __name__=='__main__':main()
