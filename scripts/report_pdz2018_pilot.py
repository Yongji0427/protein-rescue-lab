"""Evidence-calibrated result report for the PDZ point-only development pilot."""
import html
import json
import numpy as np
import pandas as pd
from pdz2018_data import OUT,dump


def main():
    p=json.loads((OUT/'protocol.json').read_text());v=json.loads((OUT/'verification.json').read_text())
    headroom=json.loads((OUT/'core_headroom_diagnosis.json').read_text())
    fixed=pd.read_csv(OUT/'fixed_runs.csv');per=fixed.groupby(['task_id','method']).success.mean().unstack()
    full=[]
    for method in p['methods']:
        values=per[method];full.append(dict(method=method,tasks=len(values),mean_successes=float(values.sum()),success_rate=float(values.mean())))
    records=[]
    for task_id in p['damage_tasks']:
        for method in p['methods']:
            row=fixed[(fixed.task_id==task_id)&(fixed.method==method)&(fixed.repeat==0)].iloc[0]
            records.append(dict(task_id=task_id,domain=row.domain,method=method,success=bool(row.success),
                best_affinity=float(row.best_affinity),queries=int(row.queries_used)))
        s=json.loads((OUT/'queried_results'/(task_id+'.json')).read_text());rows=s['queries']
        records.append(dict(task_id=task_id,domain=s['domain'],method='Agnes_AGH',success=any(r['affinity']>=.5 for r in rows),
            best_affinity=max(r['affinity'] for r in rows),queries=len(rows)))
    core=pd.DataFrame(records);core.to_csv(OUT/'core_results.csv',index=False)
    core_summary=core.groupby('method').agg(tasks=('task_id','size'),successes=('success','sum'),success_rate=('success','mean'),queries=('queries','sum')).reset_index()
    core_summary.to_csv(OUT/'core_comparison.csv',index=False)
    paired=core.pivot(index='task_id',columns='method',values='success');paired_rows=[]
    for method in ['no_feedback_contact','Agnes_AGH']:
        paired_rows.append(dict(method=method,vs='wt_rank',gains=int((paired[method]&~paired.wt_rank).sum()),losses=int((~paired[method]&paired.wt_rank).sum())))
    pd.DataFrame(paired_rows).to_csv(OUT/'paired_comparison.csv',index=False)
    changed=paired[(paired.Agnes_AGH!=paired.wt_rank)|(paired.no_feedback_contact!=paired.wt_rank)]
    changed.to_csv(OUT/'changed_tasks.csv')
    sensitivity=None
    exclude={r['task_id'] for r in v['trace_tool_deviations']}
    exclude|={r['task_id'] for r in v['completion_corrections']}
    exclude|={r['task_id'] for r in v['runtime_errors']}
    # Same exclusions for every comparator; primary result retains all scheduled trials.
    if exclude:
        sensitivity=core[~core.task_id.isin(exclude)].groupby('method').agg(tasks=('task_id','size'),successes=('success','sum'),success_rate=('success','mean')).reset_index()
        sensitivity.to_csv(OUT/'protocol_adherent_sensitivity.csv',index=False)
    result=dict(status='completed' if v['engineering_complete'] else 'completed_with_partial_model_trials',
        endpoint='point-only development; not reliable-rescue benchmark',full_fixed=full,core=core_summary.to_dict('records'),
        paired=paired_rows,core_headroom=headroom,verification=v,formal_test_status='not started; SHANK3/ZO1 remain unread',
        interpretation='The frozen GB1 structure heuristic has only a small average difference in PDZ development. Model outcomes require paired comparison on the same 32 tasks. No formal or biological improvement claim.')
    dump(OUT/'result.json',result)
    table=pd.DataFrame(full);table.success_rate=table.success_rate.map(lambda x:f'{x:.2%}')
    core_table=core_summary.copy();core_table.success_rate=core_table.success_rate.map(lambda x:f'{x:.2%}')
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>PDZ 2018 开发预实验</title>
<style>body{{font:16px/1.7 system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#173044}}h1{{font-size:29px}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #ddd}}.note{{background:#eef4f6;padding:18px}}a{{color:#176b88}}</style>
<h1>PDZ 2018 点值开发预实验</h1><p class="note">已完成数据接口、207 个开发任务固定策略回放、32 个身份哈希选定的 Agnes 任务。主要终点为公开实验结合能对应的相对结合亲和力 ≥ 0.5 WT。PDZ3 的逐突变计数置信界仍无法可靠映射，本轮不作统一可靠救援结论。SHANK3、ZO1 测试数组保持封存。</p>
<h2>全量固定策略：207 个任务</h2>{table.to_html(index=False,border=0)}
<p>WT 排序为确定性对照；结构策略使用 100 个搜索种子，平均成功任务数可以为小数。种子是搜索重复，不是生物学重复。207 个任务全部保留，其中一个只有 31 个可用候选，查询全部 31 个；其余最多 32 次。</p>
<h2>相同 32 个任务、每任务 32 次查询</h2>{core_table.to_html(index=False,border=0)}
{pd.DataFrame(paired_rows).to_html(index=False,border=0)}
<p>每个结构域按身份 SHA256 抽样（11/11/10），仅要求公开候选数超过 32，未按可救援性选任务。共有前八个 WT 排序查询，其后六批，每批四个，必须含当前 WT 锚点。Agnes 和固定策略使用相同公开菜单生成规则。</p>
<p>冻结队列的事后可救援性诊断：{headroom['point_reachable']}/32 个任务有点值救援，WT 排序找到 {headroom['WT_point32']} 个，剩余 {headroom['point_remaining']} 个搜索增益空间。该诊断未提供给模型，也未用于修改队列。</p>
<h2>核验与限制</h2><p>固定策略标签查询 {v['fixed_queries']:,} 次；模型任务实际计费标签 {v['model_queries']:,} 次（包含共享起始查询）；完成任务 {v['complete_model_tasks']}/32；回放决策批次 {v['decision_batches_replayed']:,} 个。旧 GB1 的 {v['old_GB1_hashes_unchanged']} 个冻结文件哈希保持一致。</p>
<p>模型范围偏离：{html.escape(json.dumps(v['trace_tool_deviations'],ensure_ascii=False))}。预算完成提醒：{html.escape(json.dumps(v['completion_corrections'],ensure_ascii=False))}。所有计划试验保留在主要分母中。</p>
<p>运行错误记录：{html.escape(json.dumps(v['runtime_errors'],ensure_ascii=False))}。只按冻结规则在原同会话继续，保留已查询标签和原预算；有一例在完整 32 次标签之后返回运行错误，不替换试验。</p>
{('<h3>同时排除偏离任务的敏感性比较</h3>'+sensitivity.to_html(index=False,border=0)) if sensitivity is not None else ''}
<p>结构先验保留 GB1 的化学门控、6 Å 距离核与探索比例，只替换为 PDZ 原生几何；未调参。PDZ2、Syntrophin 的条件计数敏感性不能与缺失的 PDZ3 混成统一置信结果。开发标签在前期可行性审计中已检查，本轮为开发探索，无独立验证、湿实验或机制证明。</p>
<p>AGH 初版插件因缺少显式 costHint 元数据启动失败，模型调用前修复并单独冻结；选择规则、数据和提示不变，原失败版本保留。</p>
<p><a href="protocol.json">协议</a> · <a href="verification.json">核验</a> · <a href="endpoint_diagnosis.json">评分端点诊断</a> · <a href="changed_tasks.csv">差异任务</a> · <a href="https://doi.org/10.7554/eLife.34300">数据原文</a></p></html>'''
    (OUT/'report.html').write_text(body)
    print(core_summary.to_string(index=False));print(pd.DataFrame(paired_rows).to_string(index=False))


if __name__=='__main__':main()
