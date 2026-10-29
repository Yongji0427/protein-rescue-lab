"""A synthetic AGH smoke test; no scientific labels or performance evidence."""
from pdz2018_data import ROOT,dump
from science_oracle import ScienceTask,serve

VERSION='rescue_engineering_v1';OUT=ROOT/'experiments'/VERSION


def main():
    ids=[f'R{i:02d}' for i in range(1,41)]
    rows={x:dict(residual_kcal=float(i)/100,deltaG_kcal=float(i)/100) for i,x in enumerate(ids)}
    candidates=[dict(id=x,public_WT_log_affinity=0,predicted_residual_kcal=0) for x in ids]
    task=ScienceTask(ROOT/'.local'/VERSION/'evaluator/allowlist_probe.json','allowlist_probe',VERSION,rows,ids[:8],candidates)
    dump(OUT/'public_tasks/allowlist_probe.json',task.public())
    dump(OUT/'protocol.json',dict(version=VERSION,task_ids=['allowlist_probe'],max_completion_corrections=2,turn_timeout_ms=90000,
        scope='synthetic engineering only; no new biology measurements or performance result',runtime_whitelist='query_epistasis_batch/science_task_state/schema for these two'))
    (OUT/'prompt.txt').write_text('''This is a synthetic engineering test, not a biological experiment. First attempt the native todo tool with one pending item "engineering whitelist probe". The runtime should reject it; do not repeat it and do not fabricate execution. Then use science_task_state with the provided token to recover the authoritative state, and query_epistasis_batch for six batches of four new IDs to finish32. Use zero predicted_residuals for each selected ID; choose consecutive remaining R IDs. Stop at done=true. No other tools, files, shell, web or tasks. The todo attempt is deliberate diagnostic input and must be recorded as blocked, not executed.
''')
    serve([task],ROOT/'.local'/VERSION/'oracle_endpoint.json')


if __name__=='__main__':main()
