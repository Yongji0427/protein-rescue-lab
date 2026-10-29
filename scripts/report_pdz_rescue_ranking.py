"""Analysis-only report for frozen rescue ranking runs."""
from datetime import datetime, timezone
import hashlib
import html
import json
import numpy as np
import pandas as pd
from pdz2018_data import ROOT, dump, hashes
from pdz_rescue_ranking import METHODS
from run_pdz_rescue_ranking import OUT, check_hashes

LABELS={'wt_rank':'WT 排序','contact_seed0':'原接触探索','ridge_full_lcb':'原结构残差＋全预算 LCB',
        'conservative_lcb':'① 保守预算＋LCB','conservative_pi':'② 保守预算＋阈值概率',
        'rank_pi':'③ 排序训练＋阈值概率','ensemble_pi':'④ 等权专家融合'}


def main():
    check_hashes()
    verification=json.loads((OUT/'verification.json').read_text());assert verification['status']=='passed'
    assert not (OUT/'result.json').exists(),'Do not overwrite completed analysis'
    runs=pd.read_csv(OUT/'runs.csv');metrics=pd.read_csv(OUT/'prediction_metrics.csv')
    summary=[];pairs=[]
    for scheme in ['site','domain']:
        frame=runs[runs.scheme==scheme];wt=frame[frame.method=='wt_rank'].set_index('task_id')
        for method in METHODS:
            f=frame[frame.method==method].set_index('task_id').loc[wt.index]
            gains=f.success32 & ~wt.success32;losses=~f.success32 & wt.success32
            summary.append(dict(scheme=scheme,method=method,tasks=len(f),successes=int(f.success32.sum()),rate=float(f.success32.mean()),
                success8=int(f.success8.sum()),success16=int(f.success16.sum()),success24=int(f.success24.sum()),
                gains=int(gains.sum()),losses=int(losses.sum()),net=int(gains.sum()-losses.sum()),
                gained_sites=int(f.loc[gains,'damage_site'].nunique()),mean_penalized_batch_cost=float(f.penalized_cost.mean())))
        comparisons=[('conservative_lcb','ridge_full_lcb'),('conservative_pi','conservative_lcb'),
                     ('rank_pi','conservative_pi'),('ensemble_pi','conservative_pi'),('ensemble_pi','rank_pi')]
        for candidate,control in comparisons:
            a=frame[frame.method==candidate].set_index('task_id').loc[wt.index]
            b=frame[frame.method==control].set_index('task_id').loc[wt.index]
            delta=a.success32.astype(int)-b.success32.astype(int)
            # Paired grouped sensitivity interval, not independent confirmatory inference.
            cluster=a.damage_site if scheme=='site' else a.domain
            groups=list(dict.fromkeys(cluster))
            sums=np.array([delta[cluster==g].sum() for g in groups]);sizes=np.array([(cluster==g).sum() for g in groups])
            rng=np.random.default_rng(20261008);draw=rng.integers(0,len(groups),(5000,len(groups)))
            rates=sums[draw].sum(1)/sizes[draw].sum(1)
            lo,hi=np.quantile(rates,[.025,.975])
            pairs.append(dict(scheme=scheme,candidate=candidate,control=control,gains=int((delta>0).sum()),losses=int((delta<0).sum()),
                net=int(delta.sum()),rate_difference=float(delta.mean()),cluster_count=len(groups),
                grouped_sensitivity_low=float(lo),grouped_sensitivity_high=float(hi)))
    sf=pd.DataFrame(summary);sf.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(pairs).to_csv(OUT/'stage_comparisons.csv',index=False)
    domain=runs.groupby(['scheme','domain','method'],sort=False).agg(tasks=('task_id','size'),successes=('success32','sum')).reset_index()
    domain.to_csv(OUT/'domain_results.csv',index=False)
    pred=metrics.groupby(['scheme','model']).agg(MAE=('MAE','mean'),Brier=('Brier','mean')).reset_index()
    pred.to_csv(OUT/'prediction_summary.csv',index=False)
    primary=sf[sf.scheme=='site'];transfer=sf[sf.scheme=='domain']
    new=primary[primary.method.isin(METHODS[3:])].copy();new['order']=new.method.map({m:i for i,m in enumerate(METHODS)})
    selected=new.sort_values(['successes','mean_penalized_batch_cost','order'],ascending=[False,True,True]).iloc[0]
    method=selected.method;transfer_chosen=transfer[transfer.method==method].iloc[0]
    bestref=int(primary[primary.method.isin(METHODS[:3])].successes.max())
    besttransfer=int(transfer[transfer.method.isin(METHODS[:3])].successes.max())
    dt=domain[domain.scheme=='domain'].pivot(index='domain',columns='method',values='successes')
    domain_delta=(dt[method]-dt['wt_rank']).to_dict()
    checks=dict(site_gain_vs_best_reference=bool(selected.successes>=bestref+3),multiple_gained_sites=bool(selected.gained_sites>=2),
                transfer_gain_vs_best_reference=bool(transfer_chosen.successes>=besttransfer+1),
                no_large_transfer_domain_loss=bool(min(domain_delta.values())>=-1))
    passed=all(checks.values())
    reach=int(runs[(runs.scheme=='site')&(runs.method=='wt_rank')].reachable.sum())
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Exploratory development, point-only',
        selected_method=method,selected_successes=int(selected.successes),best_reference_successes=bestref,
        transfer_selected_successes=int(transfer_chosen.successes),transfer_best_reference_successes=besttransfer,
        transfer_domain_delta_vs_WT={k:int(v) for k,v in domain_delta.items()},tasks=207,reachable=reach,
        gate_checks=checks,agnes_gate_passed=passed,stage5='eligible_pending_new_frozen_model_run' if passed else 'not_launched_failed_prerequisite',
        new_agnes_calls=0,sealed_domains_loaded=False,summary=summary,stage_comparisons=pairs,
        limitations=['Previously inspected development labels; no new independent test',
            'Only3 domains and9 scanned sites; leave-domain-out is development transfer sensitivity',
            'OOF Gaussian bias/scale calibration transfers from WT trajectories to adaptive queries; outer Brier reported, not guaranteed calibration',
            'No biological confidence endpoint; PDZ3 measurement uncertainty still unavailable',
            'Batch-first-hit costs charge the entire committed batch; seed repeats are not biological replicates',
            'Shared genotypes can couple different damage sites; grouped intervals are descriptive sensitivity only'])
    dump(OUT/'result.json',result)
    dump(OUT/'stage5_decision.json',{k:result[k] for k in ['selected_method','selected_successes','best_reference_successes',
        'transfer_selected_successes','transfer_best_reference_successes','gate_checks','agnes_gate_passed','stage5','new_agnes_calls','sealed_domains_loaded']})
    tables=[]
    for scheme,title in [('site','按损伤位点隔离：主开发比较'),('domain','留一开发域：迁移检查')]:
        f=sf[sf.scheme==scheme].copy();f['method']=f.method.map(LABELS);f['rate']=f.rate.map(lambda x:f'{100*x:.2f}%')
        f=f[['method','successes','rate','success8','success16','success24','gains','losses','mean_penalized_batch_cost']]
        f.columns=['方法','成功@32','成功率','成功@8','成功@16','成功@24','较WT新增','较WT丢失','失败惩罚后平均批成本']
        tables.append(f'<h2>{title}</h2>'+f.to_html(index=False,border=0,float_format=lambda x:f'{x:.2f}'))
    pair_frame=pd.DataFrame(pairs);pair_frame['candidate']=pair_frame.candidate.map(LABELS);pair_frame['control']=pair_frame.control.map(LABELS)
    decision='满足进入新 Agnes 对比的条件；尚未执行。' if passed else '未满足预先冻结的 Agnes 进入条件，本轮不新增模型调用。'
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>蛋白救援：排序与专家融合实验</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1280px;margin:48px auto;padding:0 28px;color:#18312e;background:#f8faf9}}h1{{font-size:30px}}h2{{margin-top:34px;font-size:21px}}table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #dce5e0;text-align:left}}th{{background:#edf3ef}}.card{{background:#e8f2ec;padding:20px;border-left:5px solid #357958}}code{{background:#e9eeeb;padding:3px}}p{{max-width:1000px}}</style>
<h1>救援排序、预算分配与轻量专家融合</h1><p>新版本 pdz2018_rescue_ranking_v1 · 开发性回放 · 207 个任务 · 32 次查询 · 共同前 8 次 WT</p>
<div class="card"><b>选中的新方法：{html.escape(LABELS[method])}</b><br>位点隔离 {int(selected.successes)}/207；最强原对照 {bestref}/207。留域 {int(transfer_chosen.successes)}/207；最强原对照 {besttransfer}/207。<br>{decision}</div>
<p>当前测量候选池可达任务 {reach}/207，上限 {100*reach/207:.2f}%。所有不可达任务继续留在主分母。没有读取 SHANK3/ZO1，没有修改旧实验。</p>
{''.join(tables)}<h2>逐步配对比较</h2>{pair_frame.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}')}
<h2>各开发域</h2>{domain.to_html(index=False,border=0)}<h2>相同前 8 次反馈后的预测误差</h2>{pred.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}')}
<h2>验证与解释</h2><p>独立重建 24 个外层模型；逐批因果重放 {verification['causal_batches_replayed']} 批、{verification['logged_queries']} 次查询；精确复现 {verification['old_baseline_trajectories_exactly_reproduced']} 条旧基线轨迹。两个代表性外层折还通过了排除标签扰动不变性测试。</p>
<p>阈值概率来自训练折内预测误差的高斯偏差与尺度校准，不是生物学置信度。排序模型与原 ridge 使用相同特征，融合为两个概率的等权平均。没有参数扫描或事后按任务挑选专家。</p>
<p>当前结果属于已查看过数据的开发探索；留域结果也是开发迁移检查。仅三个域与九个位点，分组区间不能作为独立确认性证据。PDZ3 测量不确定性缺失仍未解决。</p>
<p>详见 <a href="protocol.json">冻结协议</a>、<a href="result.json">完整结果</a>、<a href="verification.json">验证记录</a>、<a href="stage5_decision.json">Agnes 进入判定</a>。</p></html>'''
    (OUT/'report.html').write_text(body)
    dump(OUT/'analysis_manifest.json',dict(source_hashes=hashes([ROOT/'scripts/report_pdz_rescue_ranking.py',OUT/'runs.csv',OUT/'prediction_metrics.csv',OUT/'verification.json'])))
    dump(OUT/'final_artifact_hashes.json',hashes([p for p in OUT.rglob('*') if p.is_file() and p.name not in ['final_artifact_hashes.json','execution.log','verification.log']]))
    print(sf.to_string(index=False));print(json.dumps({k:result[k] for k in ['selected_method','gate_checks','stage5']},indent=2))


if __name__=='__main__':main()
