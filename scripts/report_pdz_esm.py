"""Frozen analysis for full-construct ESM feature experiment."""
from datetime import datetime,timezone
import json
import numpy as np
import pandas as pd
from pdz2018_data import dump,hashes,ROOT
from pdz_esm_model import OUT,METHODS,FUSION
from run_pdz_esm import check

LABELS={'wt_rank':'WT 排序','contact_seed0':'原接触探索','ridge_full_lcb':'原结构模型，全预算 LCB',
        'conservative_pi':'原结构模型，保守阈值概率','esm_full_lcb':'ESM增强，全预算 LCB',
        'esm_pi':'ESM增强，保守阈值概率',FUSION:'原结构＋ESM，等权融合'}


def main():
    check();assert not (OUT/'result.json').exists()
    verification=json.loads((OUT/'verification.json').read_text());assert verification['status']=='passed'
    f=pd.read_csv(OUT/'runs.csv');methods=list(dict.fromkeys(f.method));rows=[];paired=[]
    for scheme in ['site','domain']:
        frame=f[f.scheme==scheme];wt=frame[frame.method=='wt_rank'].set_index('task_id')
        for method in methods:
            a=frame[frame.method==method].set_index('task_id').loc[wt.index];gain=a.success32 & ~wt.success32;loss=~a.success32 & wt.success32
            rows.append(dict(scheme=scheme,method=method,tasks=len(a),successes=int(a.success32.sum()),success_rate=float(a.success32.mean()),
                success8=int(a.success8.sum()),success16=int(a.success16.sum()),success24=int(a.success24.sum()),gains=int(gain.sum()),losses=int(loss.sum()),
                gained_sites=int(a.loc[gain,'damage_site'].nunique()),mean_penalized_batch_cost=float(a.penalized_cost.mean())))
        comparisons=[('esm_full_lcb','ridge_full_lcb'),('esm_pi','conservative_pi')]
        if FUSION in methods:comparisons += [(FUSION,'conservative_pi'),(FUSION,'esm_pi')]
        for candidate,control in comparisons:
            a=frame[frame.method==candidate].set_index('task_id').loc[wt.index];b=frame[frame.method==control].set_index('task_id').loc[wt.index]
            delta=a.success32.astype(int)-b.success32.astype(int);cluster=a.damage_site if scheme=='site' else a.domain;groups=list(dict.fromkeys(cluster))
            sums=np.array([delta[cluster==g].sum() for g in groups]);sizes=np.array([(cluster==g).sum() for g in groups])
            rng=np.random.default_rng(20261008);idx=rng.integers(0,len(groups),(5000,len(groups)));rates=sums[idx].sum(1)/sizes[idx].sum(1)
            lo,hi=np.quantile(rates,[.025,.975])
            paired.append(dict(scheme=scheme,candidate=candidate,control=control,gains=int((delta>0).sum()),losses=int((delta<0).sum()),net=int(delta.sum()),
                rate_difference=float(delta.mean()),grouped_sensitivity_low=float(lo),grouped_sensitivity_high=float(hi),clusters=len(groups)))
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'summary.csv',index=False);pd.DataFrame(paired).to_csv(OUT/'paired_comparisons.csv',index=False)
    domain=f.groupby(['scheme','domain','method'],sort=False).agg(tasks=('task_id','size'),successes=('success32','sum')).reset_index();domain.to_csv(OUT/'domain_results.csv',index=False)
    pred=pd.read_csv(OUT/'prediction_metrics.csv').groupby(['scheme','model']).agg(MAE=('MAE','mean'),Brier=('Brier','mean')).reset_index();pred.to_csv(OUT/'prediction_summary.csv',index=False)
    siteframe=summary[summary.scheme=='site'].set_index('method');transfer=summary[summary.scheme=='domain'].set_index('method')
    chosen=sorted(['esm_full_lcb','esm_pi'],key=lambda k:(-siteframe.loc[k,'successes'],siteframe.loc[k,'mean_penalized_batch_cost'],k!='esm_pi'))[0]
    preselected=chosen
    # Fusion has to beat both constituent conservative experts and the selected single.
    if FUSION in methods and siteframe.loc[FUSION,'successes']>max(siteframe.loc[k,'successes'] for k in ['conservative_pi','esm_pi',chosen]):chosen=FUSION
    reference_site=int(siteframe.loc[METHODS[:4],'successes'].max());reference_domain=int(transfer.loc[METHODS[:4],'successes'].max())
    dt=domain[domain.scheme=='domain'].pivot(index='domain',columns='method',values='successes');deltas=dt[chosen]-dt.wt_rank
    checks=dict(site_gain_vs_best_reference=bool(siteframe.loc[chosen,'successes']>=reference_site+3),
        transfer_gain_vs_best_reference=bool(transfer.loc[chosen,'successes']>=reference_domain+1),
        multiple_gained_sites=bool(siteframe.loc[chosen,'gained_sites']>=2),no_large_transfer_domain_loss=bool(deltas.min()>=-1))
    passed=all(checks.values());fusion_gate=json.loads((OUT/'fusion_gate.json').read_text())
    result=dict(scope='Exploratory point-only development',selected_method=chosen,pre_fusion_selected_method=preselected,
        site_successes=int(siteframe.loc[chosen,'successes']),transfer_successes=int(transfer.loc[chosen,'successes']),
        best_reference_site=reference_site,best_reference_transfer=reference_domain,gate_checks=checks,agnes_gate_passed=passed,
        stage5='eligible_pending_new_frozen_model_run' if passed else 'not_launched_failed_prerequisite',
        fusion_gate=fusion_gate,summary=rows,paired_comparisons=paired,new_agnes_calls=0,sealed_domains_loaded=False,
        sequence_context='Complete deposited PDB constructs;9scan sites exact; full assay plasmid context unconfirmed',
        feature_extraction=json.loads((OUT/'embedding_manifest.json').read_text()),
        limits=['Development data previously inspected; leave-domain-out is development sensitivity, not independent formal test',
            'Only3 domains and9 positions; common genotypes couple damage tasks; grouped bootstrap is descriptive',
            'Frozen ESM2-35M with fixed128D projection per block; negative result cannot exclude larger or other PLMs',
            'Point affinity only; no biological confidence or new wet-lab evidence'])
    dump(OUT/'result.json',result);dump(OUT/'stage5_decision.json',{k:result[k] for k in ['selected_method','site_successes','transfer_successes','best_reference_site','best_reference_transfer','gate_checks','agnes_gate_passed','stage5','new_agnes_calls','sealed_domains_loaded']})
    tables=[]
    for scheme,title in [('site','位点隔离：主开发比较'),('domain','留一开发域：迁移检查')]:
        table=summary[summary.scheme==scheme].drop(columns=['scheme','tasks']).copy();table['method']=table.method.map(LABELS);table['success_rate']=table.success_rate.map(lambda p:f'{p*100:.2f}%')
        tables.append(f'<h2>{title}</h2>'+table.to_html(index=False,border=0,float_format=lambda p:f'{p:.3f}'))
    explanation='满足进入新 Agnes 比较的预设条件，尚需执行。' if passed else '未满足进入 Agnes 的预设条件，本轮不增加 Agnes 调用。'
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ESM蛋白救援开发实验</title>
<style>body{{font:16px/1.6 system-ui;max-width:1280px;margin:40px auto;padding:0 25px;background:#f7faf8;color:#18372d}}table{{border-collapse:collapse;background:white;font-size:13px;width:100%}}td,th{{border-bottom:1px solid #dce5df;padding:9px;text-align:left;overflow-wrap:anywhere}}th{{background:#e9f1eb}}h2{{margin-top:32px;font-size:21px}}.card{{padding:20px;background:#e4f0e8;border-left:5px solid #357658}}</style>
<h1>完整结构域参考上下文：ESM-2救援实验</h1><p>ESM-2 35M · 冻结表征＋小型残差预测头 · 每任务32次查询 · 全部207任务</p>
<div class="card"><b>{LABELS[chosen]}</b><br>位点隔离 {result['site_successes']}/207，原最佳 {reference_site}/207；留域 {result['transfer_successes']}/207，原最佳 {reference_domain}/207。<br>{explanation}</div>
<p>表征来自39,504条设计序列，无标签筛选，包含WT、全部单突变和全部双突变。完整PDB构建体长度95/119/87；扫描九位点完全匹配，完整实验载体序列尚未确认。新特征384维，来自全序列、突变位置及非加和表示，每块用事前固定随机投影压至128维。</p>
{''.join(tables)}<h2>配对比较</h2>{pd.DataFrame(paired).to_html(index=False,border=0,float_format=lambda p:f'{p:.4f}')}
<h2>各开发域</h2>{domain.to_html(index=False,border=0)}<h2>相同前8次查询后的预测指标</h2>{pred.to_html(index=False,border=0,float_format=lambda p:f'{p:.5f}')}
<h2>验证及证据范围</h2><p>12个外层预测头独立重建；CPU重新提取12个代表性表示，核对MPS缓存；逐批重放{verification['causal_batches_replayed']}批查询，准确复现{verification['old_trajectories_exactly_reproduced']}条原对照轨迹。位点0及PDZ2留域通过排除标签扰动不变性检查。</p>
<p>结果为已观察过数据的开发探索，留域结果不是独立测试。仅三个开发域，不能把分组区间当作生物重复。未读取SHANK3/ZO1。概率是模型校准输出，不是生物置信度。初始v1在表示提取前因投影float64/MPS不兼容停止；v2仅修复类型，保留失败记录，没有依据模型成绩修改参数。</p>
<p>参考<a href="protocol.json">冻结协议</a>、<a href="result.json">全部结果</a>、<a href="verification.json">验证</a>、<a href="sequence_mapping.json">序列映射</a>、<a href="stage5_decision.json">下一阶段判定</a>。</p></html>'''
    (OUT/'report.html').write_text(body)
    dump(OUT/'final_artifact_hashes.json',hashes([p for p in OUT.rglob('*') if p.is_file() and p.name not in ['final_artifact_hashes.json'] and p.suffix!='.log']))
    print(summary.to_string(index=False));print(json.dumps({k:result[k] for k in ['selected_method','gate_checks','stage5']},indent=2))


if __name__=='__main__':main()
