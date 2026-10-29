"""Fault-injection tests for lost responses, restart, budget and session boundaries."""
import copy
import tempfile
from pathlib import Path
from science_oracle import ScienceTask


def main():
    ids=[f'R{i:02d}' for i in range(1,41)];rows={x:dict(residual_kcal=i/100) for i,x in enumerate(ids)}
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'state.json';t=ScienceTask(path,'probe','test',rows,ids[:8],[dict(id=x) for x in ids])
        payload=dict(task_token=t.state['task_token'],session_key='test-probe',action='query',rescue_ids=ids[8:12],predicted_residuals=[0]*4,request_id='a'*64,rationale='before response loss')
        for edit in [dict(session_key='other'),dict(rescue_ids=['EVAL_ONLY']+ids[9:12]),dict(rescue_ids=[ids[8]]*4),dict(predicted_residuals=[float('nan')]*4)]:
            before=copy.deepcopy(t.state)
            try:t.handle({**payload,**edit});raise AssertionError('Invalid request accepted')
            except ValueError:pass
            assert t.state==before
        assert t.handle(payload)['new_queries_charged']==4
        # Simulate lost HTTP response: identical retry does not consume four more labels.
        assert t.handle(payload)['new_queries_charged']==0 and len(t.state['queries'])==12
        resumed=ScienceTask(path,'probe','test',rows,ids[:8],[dict(id=x) for x in ids])
        assert resumed.state==t.state
        assert resumed.handle({**payload,'request_id':'b'*64,'predicted_residuals':[9]*4})['new_queries_charged']==0
        assert resumed.state['decisions'][0]['predicted_residuals']==[0]*4
        for k,start in enumerate(range(12,32,4)):
            resumed.handle({**payload,'rescue_ids':ids[start:start+4],'request_id':str(k)*64})
        assert len(resumed.state['queries'])==32 and resumed.public()['done']
        try:resumed.handle({**payload,'rescue_ids':ids[32:36],'request_id':'z'*64});raise AssertionError('Cap exceeded')
        except ValueError:pass
        assert resumed.handle({**payload,'action':'state'})['new_queries_charged']==0
        assert {r['rescue'] for r in resumed.public()['observed']}==set(ids[:32])
    print('Passed: atomic rejection, evaluation-pool refusal, cross-session refusal, lost-response replay, restart, preserved hypotheses, 32-query cap, free state recovery')


if __name__=='__main__':main()
