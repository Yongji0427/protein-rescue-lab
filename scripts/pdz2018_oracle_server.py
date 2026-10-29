"""Budgeted PDZ evaluator; only queried labels are exposed to AGH."""
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import numpy as np
from pdz2018_data import ROOT,OUT,dump,load_development
from pdz2018_policy import PDZPolicy

PRIVATE=ROOT/'.local'/OUT.name/'evaluator'
STATE={};LOCK=threading.Lock()


def public(task):
    s=task['state']
    return dict(task_token=task['token'],task_id=s['task_id'],domain=s['domain'],damage=s['damage'],
        damage_log_affinity=s['damage_log_affinity'],WT_log_affinity=0,query_budget=32,
        queries_used=len(s['queries']),queries_remaining=32-len(s['queries']),
        rescue_threshold_affinity=.5,primary='published experimental point affinity; nine alpha2 sites only',
        observed=s['queries'],next_batch=task['menu'],done=len(s['queries'])==32)


def commit(task,ids,rationale,initial=False):
    state=task['state'];policy=task['policy'];menu=task['menu'];before=len(state['queries'])
    if before==32:raise ValueError('Task already complete')
    if not isinstance(ids,list) or len(ids)!=4 or any(not isinstance(x,str) for x in ids) or len(set(ids))!=4:
        raise ValueError('Exactly four unique current candidates required')
    if not initial:
        allowed={r['id'] for r in menu['candidates']}
        if not set(ids)<=allowed or menu['required_WT_anchor'] not in ids:raise ValueError('Respect current menu and WT anchor')
    index={r['id']:r for r in task['data']['rows']}
    if any(x not in index for x in ids) or set(ids)&{r['rescue'] for r in state['queries']}:
        raise ValueError('Candidate unavailable or already queried')
    state['decisions'].append(dict(budget_before=before,menu=menu,selected=ids,rationale=str(rationale)[:300],
        source='shared_initial_WT' if initial else 'Agnes_AGH_tool'))
    for x in ids:
        r=index[x]
        state['queries'].append(dict(query=len(state['queries'])+1,batch_cost=before+4,rescue=x,
            deltaG_kcal=r['deltaG_kcal'],log_affinity=r['log_affinity'],affinity=r['affinity'],
            count_lower_affinity=r['count_lower_affinity'],WT_rescue_log_affinity=r['WT_log_affinity'],
            source='shared_initial_WT' if initial else 'Agnes_AGH_tool'))
    policy.update(ids);task['menu']=policy.menu()
    return public(task)


def save(task):
    dump(PRIVATE/(task['state']['task_id']+'.json'),task['state'])
    dump(OUT/'queried_results'/(task['state']['task_id']+'.json'),task['state'])


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
                if token not in STATE:raise ValueError('Unknown task token')
                task=STATE[token];value=commit(task,payload.get('rescue_ids'),payload.get('rationale',''));save(task)
            status=200
        except (ValueError,TypeError,KeyError) as error:value,status={'error':str(error),'new_queries_charged':0},400
        data=json.dumps(value,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)


def main():
    p=json.loads((OUT/'protocol.json').read_text());assert not PRIVATE.exists(),'Never reset evaluator state'
    PRIVATE.mkdir(parents=True,mode=0o700)
    tasks=load_development()
    for task_id in p['damage_tasks']:
        t=tasks[task_id];domain=list(dict.fromkeys(tasks[k]['domain'] for k in tasks)).index(t['domain'])
        seed=np.random.SeedSequence([p['seed'],domain,int(t['damage'][1:-1]),ord(t['damage'][-1]),0])
        task=dict(data=t,policy=PDZPolicy(t,'no_feedback_contact',seed),token=secrets.token_hex(16),menu=None,
            state={k:t[k] for k in ['task_id','domain','damage','damage_log_affinity']})
        task['state'].update(queries=[],decisions=[])
        for _ in range(2):commit(task,[str(task['policy'].ids[i]) for i in task['policy'].proposal()],'shared initial WT',True)
        STATE[task['token']]=task;save(task);dump(OUT/'public_tasks'/(task_id+'.json'),public(task))
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);endpoint=dict(url=f'http://127.0.0.1:{server.server_port}/query')
    dump(OUT/'endpoint.json',endpoint);dump(ROOT/'.local'/OUT.name/'oracle_endpoint.json',endpoint)
    print(json.dumps(dict(ready=True,tasks=len(STATE),url=endpoint['url'])),flush=True);server.serve_forever()


if __name__=='__main__':main()
