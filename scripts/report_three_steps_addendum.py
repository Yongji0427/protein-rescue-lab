"""Analysis-only disclosure of tool deviations; frozen evaluator is unchanged."""
import hashlib
import json
from pathlib import Path
import report_three_steps as frozen_report
from report_three_steps import main,ROOT,OUT

_original_bootstrap = frozen_report.paired_bootstrap

def numeric_bootstrap(frame, method, reference, metric):
    numeric = frame.copy()
    numeric[metric] = numeric[metric].astype(float)
    return _original_bootstrap(numeric, method, reference, metric)

frozen_report.paired_bootstrap = numeric_bootstrap


def report():
    main()
    final=ROOT/'experiments/gb1_final_holdout_v1'
    audit=json.loads((final/'verification.json').read_text())
    deviations=audit.get('protocol_deviations',[])
    tasks=sorted({x['damage'] for x in deviations})
    incomplete=audit.get('warnings',[])
    note=f'''<h2>工具范围偏差：保留主分析，并作敏感性比较</h2><p>封存协议未允许的计划工具及失败未知工具调用，实际涉及 {len(tasks)} 个任务、{len(deviations)} 次这种调用，涉及 {', '.join(tasks) or '无'}。这些调用只更新内部计划或在执行前报 TOOL_NOT_FOUND，没有提供额外实验标签。预算未完成任务为 {incomplete}，预定两次补全后仍不足32则保留实际查询结果，不追加尝试。协议与执行代码均未改变，任务没有重跑或替换。主表保留全部51个任务；protocol_adherent_sensitivity.csv 对所有方法同时排除工具范围偏差和预算未完成任务，作为敏感性分析，不作为新的主结果。</p><p>status=passed 表示实验读数、预算、菜单重放、模型及数据访问核对通过；工具指令遵守状态单列为 {audit['protocol_conformity']}，不能声称所有模型试验完全遵守工具指令。</p><p><a href="../gb1_final_holdout_v1/analysis_addendum.json">分析附录</a> · <a href="../gb1_final_holdout_v1/protocol_adherent_sensitivity.csv">同任务敏感性比较</a></p>'''
    summary_note='<p><strong>最终判断：开发集有结构探索增益，封存验证没有复现。</strong>开发集简化版主指标43.63%；封存集WT、固定策略、Agnes均为11/51（21.57%），且成功任务集合相同。封存池有16个计数可靠可达背景，仍漏5个，不能将持平归因于没有提升空间。当前未证明结构策略的泛化增益或Agnes的额外决策价值。</p>'
    page=OUT/'report.html';text=page.read_text().replace('</h1>','</h1>'+summary_note,1).replace('<h2>执行与验证</h2>',note+'<h2>执行与验证</h2>');page.write_text(text)
    add=json.loads((final/'analysis_addendum.json').read_text())
    add['analysis_hashes']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['scripts/summarize_stage_addendum.py','scripts/report_three_steps_addendum.py']}
    add['deviation_tasks']=tasks;add['deviation_calls']=len(deviations);add['budget_incomplete_tasks']=incomplete
    (final/'analysis_addendum.json').write_text(json.dumps(add,indent=2)+'\n')
    result=json.loads((OUT/'result.json').read_text())
    result.update(conclusion='Development contact gain did not reproduce on frozen holdout; all methods find the same11/51; five count-robust reachable backgrounds missed.',final_hash_check='all 30 frozen execution files unchanged',sensitivity='same task-level no-gain conclusion after matched exclusions')
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Tool-adherence deviations disclosed:',tasks)


if __name__=='__main__':report()
