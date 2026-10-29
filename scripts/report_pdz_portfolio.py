"""Frozen paired analysis of candidate portfolios and nested residual heads."""
import json
import numpy as np
import pandas as pd
from pdz2018_data import dump,hashes
from pdz_portfolio_model import OUT,METHODS,PORTFOLIOS,HEADS
from run_pdz_portfolio import check
LABELS={'wt_rank':'WT排序','contact_seed0':'原接触探索','ridge_full_lcb':'原线性残差搜索','esm_full_lcb':'原ESM增强搜索',
        'portfolio3':'3WT＋1轮换专家','portfolio2':'2WT＋残差＋轮换专家','portfolio1':'WT/残差/ESM/接触各1个',
        'weighted_ridge':'损伤背景加权线性模型','tree':'普通梯度提升树','weighted_tree':'损伤背景加权梯度提升树'}


def main():
    check();assert not (OUT/'result.json').exists()
    v=json.loads((OUT/'verification.json').read_text());assert v['status']=='passed'
    f=pd.read_csv(OUT/'runs.csv');summary=[];paired=[]
    for scheme in ['site','domain']:
        frame=f[f.scheme==scheme];wt=frame[frame.method=='wt_rank'].set_index('task_id')
        for m in METHODS:
            a=frame[frame.method==m].set_index('task_id').loc[wt.index];gain=a.success32 & ~wt.success32;loss=~a.success32 & wt.success32
            summary.append(dict(scheme=scheme,method=m,tasks=len(a),successes=int(a.success32.sum()),rate=float(a.success32.mean()),
                success8=int(a.success8.sum()),success16=int(a.success16.sum()),success24=int(a.success24.sum()),
                gains=int(gain.sum()),losses=int(loss.sum()),gained_sites=int(a.loc[gain,'damage_site'].nunique()),mean_penalized_cost=float(a.penalized_cost.mean())))
        comparisons=json.loads((OUT/'protocol.json').read_text())['comparisons']
        for candidate,control in comparisons:
            a=frame[frame.method==candidate].set_index('task_id').loc[wt.index];b=frame[frame.method==control].set_index('task_id').loc[wt.index]
            delta=a.success32.astype(int)-b.success32.astype(int);cluster=a.damage_site if scheme=='site' else a.domain;groups=list(dict.fromkeys(cluster))
            sums=np.array([delta[cluster==g].sum() for g in groups]);sizes=np.array([(cluster==g).sum() for g in groups]);rng=np.random.default_rng(20261008)
            idx=rng.integers(0,len(groups),(5000,len(groups)));rate=sums[idx].sum(1)/sizes[idx].sum(1);lo,hi=np.quantile(rate,[.025,.975])
            paired.append(dict(scheme=scheme,candidate=candidate,control=control,gains=int((delta>0).sum()),losses=int((delta<0).sum()),net=int(delta.sum()),
                grouped_sensitivity_low=float(lo),grouped_sensitivity_high=float(hi),clusters=len(groups)))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'summary.csv',index=False);pd.DataFrame(paired).to_csv(OUT/'paired_comparisons.csv',index=False)
    domain=f.groupby(['scheme','domain','method'],sort=False).agg(tasks=('task_id','size'),successes=('success32','sum')).reset_index();domain.to_csv(OUT/'domain_results.csv',index=False)
    pred=pd.read_csv(OUT/'prediction_metrics.csv').groupby(['scheme','method']).MAE.mean().reset_index();pred.to_csv(OUT/'prediction_summary.csv',index=False)
    configs=[]
    for p in sorted((OUT/'selection').glob('*.json')):
        for family,c in json.loads(p.read_text())['configs'].items():configs.append(dict(fold=p.stem,method=family,**c))
    pd.DataFrame(configs).to_csv(OUT/'selected_hyperparameters.csv',index=False)
    sf=summary[summary.scheme=='site'].set_index('method');tf=summary[summary.scheme=='domain'].set_index('method')
    candidates=PORTFOLIOS+HEADS;chosen=sorted(candidates,key=lambda m:(-sf.loc[m,'successes'],sf.loc[m,'mean_penalized_cost'],METHODS.index(m)))[0]
    refsite=int(sf.loc[METHODS[:4],'successes'].max());reftransfer=int(tf.loc[METHODS[:4],'successes'].max())
    dt=domain[domain.scheme=='domain'].pivot(index='domain',columns='method',values='successes');delta=dt[chosen]-dt.wt_rank
    checks=dict(site_gain_vs_best_reference=bool(sf.loc[chosen,'successes']>=refsite+3),transfer_gain_vs_best_reference=bool(tf.loc[chosen,'successes']>=reftransfer+1),
        multiple_gained_sites=bool(sf.loc[chosen,'gained_sites']>=2),no_large_transfer_domain_loss=bool(delta.min()>=-1))
    passed=all(checks.values())
    result=dict(scope='Exploratory point-only development with nested head tuning',selected_method=chosen,site_successes=int(sf.loc[chosen,'successes']),transfer_successes=int(tf.loc[chosen,'successes']),
        best_reference_site=refsite,best_reference_transfer=reftransfer,gate_checks=checks,agnes_gate_passed=passed,
        stage5='eligible_pending_new_frozen_model_run' if passed else 'not_launched_failed_prerequisite',summary=summary.to_dict('records'),paired_comparisons=paired,
        new_agnes_calls=0,sealed_domains_loaded=False,
        limits=['Previously inspected development data','Leave-domain-out is development sensitivity, not untouched test','Only3 domains and9 sites','Point-only, not biological confidence',
          'Weighted tree and ordinary tree depths selected separately; paired difference compares nested procedures','No larger-model training or hardware change'])
    dump(OUT/'result.json',result);dump(OUT/'stage5_decision.json',{k:result[k] for k in ['selected_method','site_successes','transfer_successes','best_reference_site','best_reference_transfer','gate_checks','agnes_gate_passed','stage5','new_agnes_calls','sealed_domains_loaded']})
    tables=[]
    for scheme,title in [('site','位点隔离：主开发结果'),('domain','留一开发域：迁移检查')]:
        table=summary[summary.scheme==scheme].drop(columns=['scheme','tasks']).copy();table['method']=table.method.map(LABELS);table['rate']=table.rate.map(lambda p:f'{p*100:.2f}%')
        table.columns=['方法','成功@32','成功率','成功@8','成功@16','成功@24','较WT新增','较WT丢失','新增位点数','失败惩罚后平均批成本']
        tables.append(f'<h2>{title}</h2>'+table.to_html(index=False,border=0,float_format=lambda p:f'{p:.3f}'))
    decision='达到预设Agnes进入条件，尚待新模型运行。' if passed else '未达到预设Agnes进入条件，本轮不增加Agnes调用。'
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>救援候选组合与加权树模型</title>
<style>body{{font:16px/1.6 system-ui;max-width:1320px;margin:40px auto;padding:0 24px;background:#f7faf8;color:#193a2c}}table{{border-collapse:collapse;width:100%;font-size:13px;background:#fff}}th,td{{border-bottom:1px solid #dbe5dd;text-align:left;padding:9px;overflow-wrap:anywhere}}th{{background:#e6f0e9}}h2{{font-size:21px;margin-top:32px}}.card{{padding:20px;background:#e6f2e9;border-left:5px solid #307556}}</style>
<h1>救援候选组合、损伤加权与非线性模型</h1><p>207任务 · 共同前8次WT · 每任务最多32次 · 新方法6个 · 原对照4个</p>
<div class="card"><b>选中：{LABELS[chosen]}</b><br>位点隔离 {result['site_successes']}/207，原最佳 {refsite}/207；留域 {result['transfer_successes']}/207，原最佳 {reftransfer}/207。<br>{decision}</div>
<p>三个候选组合分别保留26、20、14个WT名额（完整32候选任务），其他名额由残差、ESM、接触策略独立提名。各专家共享本轮实际反馈并去重，未拼接旧轨迹。三个新预测头保持相同特征、GP和LCB搜索；损伤权重和树深通过每个外层训练折内的3组位点验证选择，未按外层成绩调参。</p>
{''.join(tables)}<h2>逐步配对差异</h2>{pd.DataFrame(paired).to_html(index=False,border=0,float_format=lambda p:f'{p:.4f}')}
<h2>各开发域</h2>{domain.to_html(index=False,border=0)}<h2>训练折内选出的配置</h2>{pd.DataFrame(configs).to_html(index=False,border=0)}
<h2>验证</h2><p>重建36个外层小模型；两个代表折重算全部内层配置并通过排除标签扰动不变性检查。重放{v['causal_batches_replayed']}批、{v['logged_queries']}次查询，精确复现{v['old_trajectories_exactly_reproduced']}条旧对照轨迹。旧结果及执行输入哈希保持不变，SHANK3/ZO1未读取。</p>
<p>此为已观察过数据的开发探索，留域检查不等于独立测试。仅三个域与九个位点；共享基因型使任务相关，分组区间仅作描述。结果基于亲和力点估计，不是生物学置信证据。普通树与加权树可能选择不同深度，不能把其差值全部归因于权重。</p>
<p><a href="protocol.json">冻结协议</a> · <a href="result.json">完整结果</a> · <a href="verification.json">验证记录</a> · <a href="stage5_decision.json">后续判定</a></p></html>'''
    (OUT/'report.html').write_text(body)
    dump(OUT/'final_artifact_hashes.json',hashes([p for p in OUT.rglob('*') if p.is_file() and p.name!='final_artifact_hashes.json' and p.suffix!='.log']))
    print(summary.to_string(index=False));print(json.dumps({k:result[k] for k in ['selected_method','gate_checks','stage5']},indent=2))


if __name__=='__main__':main()
