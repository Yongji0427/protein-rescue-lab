"""Export observed traces, original analysis code and evidence boundaries only."""
import hashlib
import json
from pathlib import Path

from prepare_science_toolbox import ROOT, OUT, PRIVATE
from science_toolbox import save


def main():
    preserved = json.loads((OUT/'preserved_hashes.json').read_text())
    changed = [name for name, digest in preserved.items()
        if not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest]
    if changed:
        raise RuntimeError('Previously existing artifact changed: ' + ', '.join(changed))
    records = []; tokens = []
    for path in sorted((OUT/'agh_events').glob('*.run.json')):
        run = json.loads(path.read_text()); task_id = run['task_id']
        phases = []
        for phase in ('operational_recovery', 'controller_completion', 'compacted_completion'):
            recovery_path = OUT/phase/f'{task_id}.json'
            if recovery_path.exists():
                phases.append(json.loads(recovery_path.read_text()))
        if phases:
            run = dict(run, initial_trial_finished=run['finished'], initial_trial_queries=run['queries'],
                finished=phases[-1]['finished'], queries=phases[-1]['after'], operational_recovery=phases)
        state = json.loads((PRIVATE/'evaluator'/f'{task_id}.json').read_text())
        if run['python_attempts'] != state['analysis_used']:
            run = dict(run, initial_trial_python_attempts=run['python_attempts'], python_attempts=state['analysis_used'])
        if bool(state['conclusion']) != run['finished']:
            run = dict(run, last_caller_reported_finished=run['finished'],
                finished=bool(state['conclusion']), late_persisted_conclusion_observed=True,
                lifecycle_note='Caller errored/timed out; trusted persisted conclusion and final native replay are authoritative.')
        tokens.append(state['task_token'])
        traces = (OUT/'agh_events'/f'{task_id}.jsonl').read_text()
        for phase in ('operational_recovery', 'controller_completion', 'compacted_completion', 'final_native_audit'):
            recovery_trace = OUT/phase/f'{task_id}.jsonl'
            if recovery_trace.exists():
                traces += recovery_trace.read_text()
        parsed = [json.loads(line) for line in traces.splitlines()]
        native = list({e.get('id', str((e.get('generation'), e['seq'], e['type']))): e for e in parsed}.values())
        calls = {e['data']['toolUseId']: e['data']['name'] for e in native if e['type'] == 'tool/call'}
        native_calls = [dict(name=e['data']['name'], seq=e['seq']) for e in native if e['type'] == 'tool/call']
        native_errors = [dict(tool=calls.get(e['data']['toolUseId']), code=e['data'].get('code'), seq=e['seq'])
            for e in native if e['type'] == 'tool/result' and e['data'].get('isError')]
        # The exact runtime denial is retained in the redacted native event log.
        runtime_denial = 'HOOK_DENIED' in traces
        analyses = []
        for n, analysis in enumerate(state['analyses'], 1):
            code_path = OUT/'analysis_code'/task_id/f'analysis_{n:02d}.py'
            code_path.parent.mkdir(parents=True, exist_ok=True)
            code_path.write_text(analysis['code'] + '\n')
            analyses.append(dict(purpose=analysis['purpose'], queries_used=analysis['queries_used'],
                code=analysis['code'], result=analysis['result'], code_file=str(code_path.relative_to(OUT))))
        actions = [c['name'] for c in native_calls]
        best = max(state['observed'], key=lambda r:r['affinity'])
        public = json.loads((OUT/'public_tasks'/f'{task_id}.json').read_text())
        single = next(r['WT_affinity'] for r in public['candidates'] if r['id'] == best['id'])
        controls = dict(WT=1., damage=public['damage_affinity'], rescue_single=single,
            damage_plus_rescue=best['affinity'], rescue_id=best['id'], damage_retained=True,
            additional_substitutions=1, point_recovery=best['affinity']>=.5,
            evidence='deterministic four-control reconstruction for the best observed candidate; not agent narrative')
        point_found = best['affinity'] >= .5
        conclusion_consistent = state['conclusion'] is not None and (
            (state['conclusion']['outcome'] == 'measured_rescue') == point_found)
        records.append(dict(run=run, observed=state['observed'], decisions=state['decisions'],
            hypotheses=state['hypotheses'], analyses=analyses, conclusion=state['conclusion'],
            tool_actions=actions, native_errors=native_errors, runtime_denial=runtime_denial,
            observed_point_check=dict(best_id=best['id'], best_affinity=best['affinity'],
                point_recovery_found=point_found, agent_outcome_consistent=conclusion_consistent, four_controls=controls),
            engineering=dict(custom_python_executed=bool(analyses),
                successful_python=any(a['result'].get('ok') and (a['result'].get('result') is not None or a['result'].get('stdout')) for a in analyses),
                hypothesis_before_new_query=any(h['queries_used']==8 for h in state['hypotheses']),
                hypothesis_after_new_query=any(h['queries_used']>8 for h in state['hypotheses']),
                real_tool_query=bool(state['decisions']), conclusion_committed=state['conclusion'] is not None)))
    offline = json.loads((OUT/'offline_verification.json').read_text())
    plugin_checks = json.loads((OUT/'plugin_verification.json').read_text())
    inventory = json.loads((OUT/'plugin_inventory.json').read_text())
    expected = set()
    for manifest in OUT.glob('*_manifest.json'):
        expected.update(json.loads(manifest.read_text()).get('task_ids', []))
    recorded = {r['run']['task_id'] for r in records}
    result = dict(status='completed' if expected == recorded and records and all(r['run']['finished'] for r in records) else 'partial',
        implementation_status='verified' if offline['status']=='passed' and plugin_checks['status']=='passed' and inventory['actual']=='running' else 'failed',
        plugin=inventory, plugin_checks=plugin_checks,
        prepared_tasks=len(expected), recorded_tasks=len(recorded), pending_task_ids=sorted(expected-recorded),
        tools=10, offline=offline, trials=[dict(**r['run'], engineering=r['engineering'],
            runtime_denial=r['runtime_denial'], tool_actions=r['tool_actions'], native_errors=r['native_errors'],
            observed_point_check=r['observed_point_check']) for r in records],
        preserved_existing_files=len(preserved), existing_files_unchanged=True,
        sealed_domains_loaded=False, evidence='toolbox execution evidence; no search improvement or independent biological validation',
        external_pretrained_predictor_connected=False,
        prediction_tools='public additive prior, observed-only ridge, custom agent Python')
    save(OUT/'result.json', result)
    data = json.dumps(records, ensure_ascii=False).replace('<', '\\u003c')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Protein Rescue Lab · 科学工具箱</title>
<style>body{font:16px/1.65 system-ui;max-width:1150px;margin:36px auto;padding:0 24px;color:#173044}h1{font-size:30px}select{font:inherit;padding:8px;max-width:100%}.box{background:#edf5f7;padding:18px;border-left:4px solid #2b7a85}.warning{background:#fff4df;padding:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f7;padding:15px;font-size:13px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #ddd;padding:9px;text-align:left}.cards{display:flex;gap:16px;flex-wrap:wrap}.card{padding:16px;background:#edf5f7}.card b{display:block;font-size:25px}</style>
<h1>Protein Rescue Lab · 科学工具箱 v1</h1>
<p class="box">Agnes 可以选择研究步骤、检查序列与公共结构、编写 Python、拟合轻量模型、记录假设、查询实验反馈，并提交有对照的结论。保留损伤，只允许增加一处补偿。</p>
<p class="warning">合成任务验证工程能力；PDZ 任务使用已检查过的开发数据回放。这里不证明智能体提高了搜索成功率，也不证明补偿机制。尚未接入外部预训练预测器。</p>
<select id="task" aria-label="选择研究任务"></select><div id="cards" class="cards"></div>
<h2>研究动作（原生调用，含错误尝试）</h2><pre id="actions"></pre><h2>假设与更新（智能体原文，未独立验证）</h2><div id="hypotheses"></div>
<h2>自主 Python 分析</h2><div id="analyses"></div><h2>反馈前提交的实验计划</h2><pre id="decisions"></pre>
<h2>已经查询的反馈</h2><div id="observed"></div><h2>确定性点值检查</h2><pre id="pointCheck"></pre><h2>智能体结论原文与四组对照</h2><pre id="conclusion"></pre>
<p><a href="result.json">执行结果</a> · <a href="offline_verification.json">隔离与预算测试</a> · <a href="protocol.json">任务协议</a> · <a href="README.md">运行说明</a></p>
<script>const records=RECORDS;const selector=document.getElementById('task');
for(const r of records){const option=document.createElement('option');option.value=r.run.task_id;option.textContent=r.run.task_id+' · '+r.run.dataset;selector.append(option)}
function pre(value){const p=document.createElement('pre');p.textContent=typeof value==='string'?value:JSON.stringify(value,null,2);return p}
function render(){const r=records.find(x=>x.run.task_id===selector.value);if(!r)return;const cards=document.getElementById('cards');cards.replaceChildren();
for(const [label,value] of [[r.run.dataset==='synthetic'?'合成反馈查询':'历史测量查询',r.run.queries+'/32'],['自主分析尝试',r.analyses.length],['假设记录',r.hypotheses.length],['结论已提交',r.run.finished?'是':'否']]){const c=document.createElement('div');c.className='card';c.textContent=label;const b=document.createElement('b');b.textContent=value;c.append(b);cards.append(c)}
document.getElementById('actions').textContent=r.tool_actions.join(' → ');document.getElementById('hypotheses').replaceChildren(...r.hypotheses.map(pre));
document.getElementById('analyses').replaceChildren(...r.analyses.flatMap(a=>[pre({purpose:a.purpose,queries_used:a.queries_used,result:a.result}),pre(a.code)]));
document.getElementById('decisions').textContent=JSON.stringify(r.decisions,null,2);document.getElementById('conclusion').textContent=JSON.stringify(r.conclusion,null,2);
document.getElementById('pointCheck').textContent=JSON.stringify(r.observed_point_check,null,2);
const table=document.createElement('table');const header=document.createElement('tr');for(const name of ['补偿',r.run.dataset==='synthetic'?'合成反馈值':'历史相对亲和力','揭示前预测','来源']){const th=document.createElement('th');th.textContent=name;header.append(th)}table.append(header);
for(const row of r.observed){const tr=document.createElement('tr');for(const v of [row.id,row.affinity.toFixed(5),row.prereveal_prediction??'共享初始测量',row.source]){const td=document.createElement('td');td.textContent=v;tr.append(td)}table.append(tr)}document.getElementById('observed').replaceChildren(table)}
selector.addEventListener('change',render);render();</script></html>'''.replace('RECORDS', data)
    (OUT/'report.html').write_text(document)
    exported = [p for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl', '.html', '.py')]
    leaks = [str(p.relative_to(OUT)) for p in exported if any(token in p.read_text() for token in tokens)]
    if leaks:
        raise RuntimeError('Task token present in export: ' + ', '.join(leaks))
    save(OUT/'verification.json', dict(status='passed', offline=offline,
        old_files_verified=len(preserved), old_files_unchanged=True, exported_task_tokens=0,
        native_trials=len(records), completed_trials=sum(r['run']['finished'] for r in records),
        engineering_checks=[dict(task_id=r['run']['task_id'], **r['engineering']) for r in records],
        limitation='Existing-file hash check plus engineering tests; not a scientific gain evaluator'))
    inputs = [p for p in (ROOT/'plugins/science-toolbox').glob('*') if p.is_file()]
    inputs += [ROOT/'scripts'/name for name in ['science_toolbox.py', 'toolbox_python.py', 'prepare_science_toolbox.py',
        'run_science_toolbox.mjs', 'install_science_toolbox.mjs', 'toolbox_agh_client.mjs', 'test_science_toolbox.py',
        'test_science_toolbox.mjs', 'recover_science_toolbox.mjs', 'audit_science_toolbox.mjs',
        'prepare_toolbox_runtime.py', 'report_science_toolbox.py']]
    inputs += [OUT/'protocol.json', OUT/'prompt.txt']
    save(OUT/'implementation_hashes.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    print(json.dumps(dict(status=result['status'], trials=len(records), old_files_unchanged=True, report=str(OUT/'report.html'))))


if __name__ == '__main__':
    main()
