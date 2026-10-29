"""Local interactive trace viewer; displays only already queried measurements."""
import json
from pdz2018_data import ROOT,dump


def main():
    old=ROOT/'experiments/pdz2018_point_dev_v1';out=ROOT/'experiments/rescue_engineering_v1'
    v=json.loads((old/'verification.json').read_text());data=[]
    for path in sorted((old/'queried_results').glob('*.json')):
        s=json.loads(path.read_text());r=json.loads((old/'agh_events'/(path.stem+'.run.json')).read_text())
        data.append(dict(task_id=path.stem,domain=s['domain'],damage=s['damage'],queries=s['queries'],
            decisions=[dict(budget_before=d['budget_before'],selected=d['selected'],rationale=d['rationale']) for d in s['decisions']],
            complete=r['complete'],turns=[dict(attempt=t['attempt'],before=t['before'],after=t['after'],error=t.get('outcome',{}).get('error')) for t in r['turns']],
            deviations=[d for d in v['trace_tool_deviations'] if d['task_id']==path.stem]))
    safe=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
    body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Protein Rescue Lab · 轨迹审计</title>
<style>body{font:16px/1.6 system-ui;max-width:1150px;margin:35px auto;padding:0 24px;color:#173044}select,button{font:inherit;padding:8px;margin-right:12px}.cards{display:flex;gap:20px;flex-wrap:wrap}.card{padding:18px;background:#eef4f6;min-width:180px}.card strong{display:block;font-size:28px}table{border-collapse:collapse;width:100%;margin-top:18px}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}.warn{color:#a44916}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f8;padding:15px}h1{font-size:28px}</style>
<h1>Protein Rescue Lab · 查询轨迹审计</h1><p>历史 PDZ 点值开发实验：32/32 完成预算，三种方法均 9/32；此视图展示已查询标签和操作恢复，不改算法成绩。历史范围偏离及错误原样保留，新运行时拒绝非白名单工具。</p>
<select id="tasks" aria-label="选择实验任务"></select><button id="problem">查看异常示例</button><div id="cards" class="cards"></div>
<h2>恢复与工具范围</h2><pre id="status"></pre><h2>逐批决策（反馈前提交）</h2><div id="decisions"></div>
<h2>已查询的实验反馈</h2><p>前8次是共享 WT 查询。历史 count lower 仅为旧固定校准计数敏感性，不是生物学置信界，也不是新整库校正端点。未经查询的标签不在此页面中。</p><div id="queries"></div>
<p><a href="verification.json">新工程核验</a> · <a href="../pdz2018_point_dev_v1/report.html">原科学结果</a></p>
<script>const records=DATA;const select=document.getElementById('tasks');
for(const r of records){const o=document.createElement('option');o.value=r.task_id;o.textContent=r.task_id;select.append(o)}
function table(target,heads,rows){const t=document.createElement('table');const header=document.createElement('tr');for(const h of heads){const c=document.createElement('th');c.textContent=h;header.append(c)}t.append(header);for(const row of rows){const tr=document.createElement('tr');for(const v of row){const td=document.createElement('td');td.textContent=String(v??'不可用');tr.append(td)}t.append(tr)}target.replaceChildren(t)}
function render(){const r=records.find(x=>x.task_id===select.value);const cards=document.getElementById('cards');cards.replaceChildren();for(const [label,val] of [['已计费查询',r.queries.length+'/32'],['实际点值成功',r.queries.some(x=>x.affinity>=.5)?'是':'否'],['运行回合',r.turns.length],['范围偏离',r.deviations.length]]){const c=document.createElement('div');c.className='card';const a=document.createElement('span');a.textContent=label;const b=document.createElement('strong');b.textContent=val;c.append(a,b);cards.append(c)}
document.getElementById('status').textContent=JSON.stringify({complete:r.complete,turns:r.turns,deviations:r.deviations},null,2);
table(document.getElementById('decisions'),['批次前预算','所选组合','提交理由'],r.decisions.map(d=>[d.budget_before,d.selected.join(', '),d.rationale]));
table(document.getElementById('queries'),['查询','批计费至','R','相对WT亲和力','历史计数下界','来源'],r.queries.map(q=>[q.query,q.batch_cost,q.rescue,q.affinity.toFixed(5),q.count_lower_affinity?.toFixed(5),q.source]));}
select.addEventListener('change',render);document.getElementById('problem').addEventListener('click',()=>{select.value='syntrophin__A4F';render()});render();</script></html>'''.replace('DATA',safe)
    (out/'dashboard.html').write_text(body)
    dump(out/'dashboard_manifest.json',dict(tasks=len(data),query_rows=sum(len(r['queries']) for r in data),
        includes_unqueried_labels=False,includes_task_tokens=False,source_experiment=old.name))
    print('Built local trace viewer:',len(data),'tasks')


if __name__=='__main__':main()
