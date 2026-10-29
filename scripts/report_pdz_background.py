"""Post-freeze analysis only; no selection tuning, oracle calls or test loading."""
import hashlib
import html
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pdz2018_data import ROOT, dump, hashes
from run_pdz_background import OUT


def main():
    result=json.loads((OUT/'result.json').read_text())
    verification=json.loads((OUT/'verification.json').read_text())
    assert verification['status']=='passed'
    runs=pd.read_csv(OUT/'runs.csv');mae=pd.read_csv(OUT/'matched_MAE.csv')
    missed=pd.read_csv(OUT/'missed_backgrounds.csv');details=pd.read_csv(OUT/'missed_rescue_details.csv')
    pivot=runs.pivot(index='task_id',columns='method',values='success')
    chosen=result['selected_method']
    changed=[]
    for task_id in pivot.index:
        if pivot.loc[task_id,chosen]!=pivot.loc[task_id,'wt_rank']:
            task=runs[(runs.task_id==task_id)&(runs.method==chosen)].iloc[0]
            control=runs[(runs.task_id==task_id)&(runs.method=='wt_rank')].iloc[0]
            changed.append(dict(task_id=task_id,change='gain' if task.success else 'loss',
                WT_best_affinity=control.best_affinity,model_best_affinity=task.best_affinity,damage_site=task.damage_site))
    pd.DataFrame(changed).to_csv(OUT/'changed_backgrounds.csv',index=False)
    domain=runs.groupby(['domain','method']).success.agg(['sum','count']).reset_index()
    domain.to_csv(OUT/'domain_results.csv',index=False)
    domain_mae=mae.groupby(['domain','model']).MAE_kcal.mean().unstack()
    domain_mae.to_csv(OUT/'domain_MAE.csv')
    site_results=runs.groupby(['damage_site','method']).success.sum().unstack()
    site_results.to_csv(OUT/'site_results.csv')
    actual=details[details.actual_Agnes_task]
    canonical={(row.domain,tuple(sorted([row.damage,row.rescue],key=lambda x:int(x[1:-1])))) for row in details.itertuples()}
    diagnostic=dict(missed_backgrounds=len(missed),true_rescue_rows=len(details),unique_double_genotypes=len(canonical),
        WT_min_rescue_rank_range=[int(missed.minimum_WT_rank.min()),int(missed.minimum_WT_rank.max())],
        rescue_rows_excluded_by_old_chemical_gate=int((~details.chemical_gate).sum()),
        hypothetical_fixed_menu_exposed_rows=int((details.fixed_menu_budgets!='[]').sum()),
        hypothetical_fixed_menu_exposed_backgrounds=int(details.loc[details.fixed_menu_budgets!='[]','task_id'].nunique()),
        actual_old_Agnes_missed_backgrounds=int(missed.actual_Agnes_task.sum()),
        actual_old_Agnes_rescue_rows=len(actual),actual_old_Agnes_exposed_rows=int((actual.actual_menu_budgets!='[]').sum()),
        actual_old_Agnes_selected_rows=int(actual.actual_selected.astype(bool).sum()),
        warning='hypothetical menus are counterfactual fixed-policy exposure, NOT real agent trajectories; 18 damage backgrounds include reverse orientations of some identical double genotypes')
    dump(OUT/'diagnosis_summary.json',diagnostic)
    assert not result['model_gate_passed'],'This report branch is for a failed prerequisite gate'
    decision=dict(at_utc=datetime.now(timezone.utc).isoformat(),stage1='completed',stage2='completed_and_verified',
        stage3='not_launched_due_to_failed_frozen_prerequisite',selected_method=chosen,
        required_net_success=3,observed_net_success=next(s['net'] for s in result['summary'] if s['method']==chosen),
        gate_checks=result['gate_checks'],new_Agnes_calls=0,new_Agnes_queries=0,
        test_labels_read=False,test_domains=['shank3','zo1'],email_sent=False,
        action='Retain WT as search control; retain ridge for development prediction evidence only. Stop this single-round method exploration; prioritize completed guarded-engineering demo. No parameter sweep, outcome-selected cohort or test unlocking.')
    dump(OUT/'stage3_decision.json',decision)
    table=pd.DataFrame(result['summary'])[['method','successes','tasks','success_rate','gains','losses','net']]
    table['success_rate']=table.success_rate.map(lambda v:f'{100*v:.2f}%')
    matched=pd.DataFrame([dict(model=k,MAE_kcal=v) for k,v in result['matched_MAE'].items()])
    matched.MAE_kcal=matched.MAE_kcal.map(lambda v:f'{v:.4f}')
    reduction=1-result['matched_MAE']['structure']/result['matched_MAE']['additive_zero']
    structure_change=result['matched_MAE']['structure']/result['matched_MAE']['sequence']-1
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PDZ 背景残差实验 v1</title>
<style>body{{font:16px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#192c35}}table{{border-collapse:collapse;margin:20px 0;width:100%;font-size:14px}}th,td{{padding:8px;border:1px solid #d7e0e4;text-align:left}}th{{background:#edf3f6}}.box{{padding:18px;background:#fff3dd;border-left:4px solid #c08825}}code{{font-size:13px}}</style>
<h1>PDZ 背景残差实验：三步执行记录</h1>
<p class="box"><b>残差可预测，但搜索优势没有通过门槛。</b>结构模型 81/207，WT 80/207；新增 5、丢失 4，净增 1。第三步按预先冻结的门槛停止，未启动新 Agnes 调用，SHANK3/ZO1 保持封存。</p>
<h2>1. 18 个遗漏背景的诊断</h2>
<p>18 个 WT@32 未命中的可达损伤背景有 23 条救援记录，来自 {len(canonical)} 个不同双突变基因型，不能把损伤/救援反向记录当作独立实验。每个背景最靠前的真救援 WT 排名为 33–125。旧化学门控排除了 {diagnostic['rescue_rows_excluded_by_old_chemical_gate']}/23 条；反事实固定规则菜单仅暴露 {diagnostic['hypothetical_fixed_menu_exposed_rows']}/23 条。</p>
<p>只有 syntrophin V5M、L8P 这两个遗漏背景实际运行过旧 Agnes。真救援 Q6M、D2Q 都从未出现在真实菜单里，模型也没有查询它们。因此这两个失败有候选暴露限制；其余 16 个背景没有真实模型轨迹，不能据此推断模型能力。WT 距离及化学门控不足以覆盖这批救援，这是描述性开发诊断。</p>
{missed.to_html(index=False,border=0)}
<h2>2. 分组模型和预算内搜索</h2>
<p>预测 ε<sub>G</sub>=G(D+R)−G(D)−G(R)，有利补偿为负，单位 kcal/mol。32,851 个有效开发双突变被去除反向重复；9 个位点折，每折排除三个开发域中任一端接触损伤位点的全部双突变。历史训练数据和当前任务候选零交叉。公开单突变允许使用，训练折单独标准化。岭回归 α=10，固定 GP 核及正则化噪声，批量先选后揭示。</p>
<p><b>访问数据范围已经改变：</b>学习方法获得位点排除后的历史开发双突变训练标签，另有每任务最多 32 次查询。故相对 WT 的差异并非纯结构或纯智能体效果。序列/结构消融则使用相同训练标签、分组、初始 8 次观测、在线规则和预算。开发标签早前已审计，本轮是探索性分组验证，未声称独立验证。</p>
{table.to_html(index=False,border=0)}
<h3>相同初始 8 条观测后的剩余全候选 MAE（任务宏平均）</h3>
{matched.to_html(index=False,border=0)}
<p>结构模型相对零残差加性预测降低 {100*reduction:.2f}% MAE；但相对序列模型 MAE 上升 {100*structure_change:.2f}%，没有达到结构增益门槛。预测误差的下降没有转化为稳定的稀疏救援搜索收益。GP 标准差是模型探索量，固定 .09 kcal² 正则化不代表生物学测量置信区间。表中成功仅为发表点能量对应亲和力 ≥0.5 WT，PDZ3 重复实验不确定性仍不可用。</p>
{domain.to_html(index=False,border=0)}
<h3>结构搜索相对 WT 的逐背景变化</h3>{pd.DataFrame(changed).to_html(index=False,border=0)}
<h2>3. Agnes 接入门槛和停止决定</h2>
<p>冻结门槛要求净增至少 3 个成功背景、收益覆盖至少 2 个损伤位点、MAE 至少降低 5%。本轮净增 1，因此未通过。没有为了完成流程而扩大阳性队列、修改阈值或解封测试集。没有新增 Agnes 调用；旧结果未改。若以后重开算法研究，应先单独预注册“保护 WT 已知强排序、探索低排名补偿”的候选策略，并在开发集证明收益后再考虑智能体；本轮不执行这项追加实验。</p>
<h2>复核和交付</h2>
<p>独立源数据复核 32,851 个历史基因型；9,936 个因果批次重放；39,738 条分方法预算查询记录；任一任务最多 32（一个池为 31），各臂初始 8 条一致。排除位点标签大幅扰动不能改变对应折模型。原 GB1、PDZ 执行及结果哈希均保留，未加载任何封存域数组。</p>
<p>方法参考：<a href="https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification">岭回归</a>；<a href="https://gaussianprocess.org/gpml/chapters/RW2.pdf">高斯过程回归</a>。本实验实现固定、无参数搜索。</p>
<p>协议：protocol.json；冻结：freeze.json；诊断：missed_rescue_details.csv；逐任务：runs.csv；查询：query_log.csv.gz；复核：verification.json；第三步决定：stage3_decision.json。</p></html>'''
    (OUT/'report.html').write_text(content)
    print(json.dumps(dict(diagnosis=diagnostic,stage3=decision),indent=2),flush=True)


if __name__=='__main__':main()
