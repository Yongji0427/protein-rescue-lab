"""Posthoc diagnostic only: residualize ranks, never fit a search predictor."""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from pdz2018_data import ROOT,RT,dump,load_development,hashes
from pdz_two_checks import OUT,Scores,ordered,check


def main():
    check();assert not (OUT/'confounding_addendum.json').exists()
    tasks=load_development();score=Scores();rows=[]
    for tid,t in tasks.items():
        b=score.task(t);_,order=ordered(t,b,'wt_single');idx=np.array(order)
        epsilon=np.array([r['deltaG_kcal']+RT*(t['damage_log_affinity']+r['WT_log_affinity']) for r in t['rows']])
        x=rankdata(b['coupling_sym'][idx]);y=rankdata(-epsilon[idx]);z=rankdata(b['wt_single'][idx]);design=np.c_[np.ones(len(z)),z]
        xr=x-design@np.linalg.lstsq(design,x,rcond=None)[0];yr=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
        partial=float(np.corrcoef(xr,yr)[0,1]);c=np.corrcoef([x,y,z]);formula=(c[0,1]-c[0,2]*c[1,2])/np.sqrt((1-c[0,2]**2)*(1-c[1,2]**2))
        assert np.isclose(partial,formula,atol=1e-10)
        rank_stability=float(np.corrcoef(rankdata(b['esm_wt'][idx]),rankdata(b['esm_conditional'][idx]))[0,1])
        rows.append(dict(task_id=tid,domain=t['domain'],raw_spearman=float(c[0,1]),correlation_with_WT_single=float(c[0,2]),
            partial_rank_correlation=partial,conditional_vs_WT_esm_rank_correlation=rank_stability))
    f=pd.DataFrame(rows);f.to_csv(OUT/'confounding_by_task.csv',index=False)
    grouped=f.groupby('domain').mean(numeric_only=True).reset_index();grouped.to_csv(OUT/'confounding_by_domain.csv',index=False)
    results=pd.read_csv(OUT/'signal_tasks.csv');p=results.pivot(index='task_id',columns='method',values='potential32')
    gains=int((p.esm_conditional & ~p.esm_wt).sum());losses=int((~p.esm_conditional & p.esm_wt).sum())
    obj=dict(status='completed_verified',scope='Posthoc diagnostic after primary frozen results; primary files and gates unchanged',tasks=len(f),
        raw_macro_spearman=float(f.raw_spearman.mean()),partial_macro_spearman=float(f.partial_rank_correlation.mean()),
        per_domain=grouped.to_dict('records'),conditional_vs_WT_esm_rank_correlation_median=float(f.conditional_vs_WT_esm_rank_correlation.median()),
        conditional_gain_vs_ESM_WT=gains,conditional_loss_vs_ESM_WT=losses,
        validation='Each task residualized-rank correlation agrees with independent analytic partial-correlation formula within1e-10',
        interpretation='Most observed raw association is removed after controlling public rescue-single affinity. This check does not establish useful incremental rescue information; does not prove absence of every nonlinear signal.',
        limitations='Only rank-linear adjustment for one covariate; posthoc developmental description; no formal causal or independent-test inference',
        new_predictive_models_trained=0,new_adaptive_queries=0,sealed_domains_loaded=False)
    dump(OUT/'confounding_addendum.json',obj)
    evidence=pd.read_csv(OUT/'evidence_summary.csv').query("run=='pdz2018_portfolio_v1' and scheme=='site' and method in ['wt_rank','ridge_full_lcb','tree','portfolio2']")
    signal=pd.read_csv(OUT/'signal_summary.csv')[['method','potential32','count_supported','full_count_supported']]
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>两项检查：最终解读</title><style>body{{font:16px/1.7 system-ui;max-width:1080px;margin:40px auto;padding:0 24px;color:#19382c}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#eef5ef;padding:18px}}h2{{margin-top:32px}}</style>
<h1>证据检查与条件补偿信号：最终解读</h1><p class="note">两项检查均完成并验证。可靠性资料缺口尚未补齐；当前35M条件评分未显示可用于搜索的增量收益。以下偏相关为事后诊断，原冻结评分与通过判定均保留。</p>
<h2>同一140个可计算计数界的任务</h2>{evidence[['method','point_success_available','count_supported','full_count_supported']].to_html(index=False,border=0)}
<p>WT找到全部10个组合计数支持任务，其中8个任务的损伤单突变也得到计数确认。两个支持端点均是固定校准下的条件计数敏感性，不是生物学置信区间。67个PDZ3任务的该端点仍缺失。</p>
<h2>固定前8个WT候选加上剩余前24名的静态覆盖</h2>{signal.to_html(index=False,border=0)}<p>此表为静态潜在覆盖，不是新的主动搜索实验。所有方法使用相同207任务与候选池。没有训练预测头或改动搜索。</p>
<h2>新评分是否提供独立信号</h2><p>补偿分数与有利能量残差的任务平均Spearman为 {obj['raw_macro_spearman']:.4f}。控制公开救援单突变强弱后，平均偏秩相关为 <b>{obj['partial_macro_spearman']:.4f}</b>。残差化计算与独立解析公式逐任务一致。</p>
{grouped.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}')}
<p>ESM条件评分与ESM的WT背景评分，其候选排名相关的任务中位数为 {obj['conditional_vs_WT_esm_rank_correlation_median']:.4f}；条件评分相对ESM-WT在最终静态覆盖上新增 {gains} 个、丢失 {losses} 个。原始正相关不足以作为升级搜索方法的依据。单一协变量的线性秩调整不能排除所有非线性信号。</p>
<h2>下一步讨论依据</h2><p>当前先保留WT及保住10个计数支持任务的对照。数据侧优先取得逐重复计数、PDZ3聚合映射、校准不确定性和构建体序列，或审计独立数据的适用性。方法侧暂不把当前补偿分数接入搜索；独立MSA耦合或更强模型属于尚未完成的新假设，需要另行设计。</p>
<p><a href="report.html">原冻结完整报告</a> · <a href="result.json">冻结结果</a> · <a href="confounding_addendum.json">事后诊断</a> · <a href="verification.json">独立验证</a> · <a href="source_refresh.json">来源刷新记录</a></p></html>'''
    (OUT/'discussion.html').write_text(text)
    paths=[ROOT/'scripts/diagnose_pdz_signal_confounding.py',OUT/'final_artifact_hashes.json',OUT/'confounding_addendum.json',OUT/'confounding_by_task.csv',OUT/'confounding_by_domain.csv',OUT/'discussion.html']
    dump(OUT/'addendum_manifest.json',dict(hashes=hashes(paths),primary_results_unchanged=True))
    check();print(json.dumps(obj,indent=2))


if __name__=='__main__':main()
