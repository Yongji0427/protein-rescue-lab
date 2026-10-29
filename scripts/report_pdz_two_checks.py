"""Report evidence limits and predeclared zero-shot rescue-signal tests."""
import json
import numpy as np
import pandas as pd
from pdz2018_data import dump,hashes
from pdz_two_checks import OUT,METHODS,check
LABELS={'wt_single':'公开WT单突变排序','esm_wt':'ESM的WT背景评分','esm_conditional':'ESM损伤背景条件评分',
        'coupling_sym':'双向对称补偿评分','wt_coupling_rank':'WT与补偿分数等权排名融合'}


def main():
    check();assert not (OUT/'result.json').exists()
    v=json.loads((OUT/'verification.json').read_text());assert v['status']=='passed'
    f=pd.read_csv(OUT/'signal_tasks.csv');summary=[];paired=[]
    wt=f[f.method=='wt_single'].set_index('task_id')
    for method in METHODS:
        a=f[f.method==method].set_index('task_id').loc[wt.index];g=a.potential32&~wt.potential32;l=~a.potential32&wt.potential32
        sub=a[(~a.initial_success)&a.reachable]
        available=a[a.count_available]
        summary.append(dict(method=method,tasks=len(a),potential16=int(a.potential16.sum()),potential24=int(a.potential24.sum()),potential32=int(a.potential32.sum()),
            gains=int(g.sum()),losses=int(l.sum()),gained_sites=int(a.loc[g,'damage_site'].nunique()),available_tasks=len(available),
            count_supported=int(available.count_supported.sum()),full_count_supported=int(available.full_count_supported.sum()),
            initially_unsolved_reachable_tasks=len(sub),remaining_AP=float(sub.average_precision.mean()),remaining_MRR=float(sub.reciprocal_first_rescue.mean()),
            macro_spearman=float(a.spearman_favorable_epsilon.mean()) if a.spearman_favorable_epsilon.notna().any() else None))
        delta=a.potential32.astype(int)-wt.potential32.astype(int);groups=sorted(a.damage_site.unique());sums=np.array([delta[a.damage_site==g].sum() for g in groups]);sizes=np.array([(a.damage_site==g).sum() for g in groups]);rng=np.random.default_rng(20261008)
        draws=rng.integers(0,len(groups),(5000,len(groups)));rates=sums[draws].sum(1)/sizes[draws].sum(1);lo,hi=np.quantile(rates,[.025,.975])
        paired.append(dict(method=method,net=int(delta.sum()),rate_difference=float(delta.mean()),grouped_sensitivity_low=float(lo),grouped_sensitivity_high=float(hi)))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'signal_summary.csv',index=False);pd.DataFrame(paired).to_csv(OUT/'paired_signal.csv',index=False)
    domain=f.groupby(['domain','method'],sort=False).agg(tasks=('task_id','size'),potential32=('potential32','sum'),macro_spearman=('spearman_favorable_epsilon','mean')).reset_index();domain.to_csv(OUT/'signal_by_domain.csv',index=False)
    sf=summary.set_index('method');choices=['esm_conditional','coupling_sym','wt_coupling_rank'];chosen=sorted(choices,key=lambda m:(-sf.loc[m,'potential32'],-sf.loc[m,'count_supported'],choices.index(m)))[0]
    pivot=domain.pivot(index='domain',columns='method',values='potential32');net=pivot[chosen]-pivot.wt_single
    checks=dict(net_coverage=bool(sf.loc[chosen,'potential32']>=sf.loc['wt_single','potential32']+3),multiple_sites=bool(sf.loc[chosen,'gained_sites']>=2),
        multiple_domains=bool((net>0).sum()>=2),count_supported_not_reduced=bool(sf.loc[chosen,'count_supported']>=sf.loc['wt_single','count_supported']))
    corr=domain[domain.method=='coupling_sym'];association=bool(sf.loc['coupling_sym','macro_spearman']>0 and (corr.macro_spearman>0).sum()>=2)
    evidence=pd.read_csv(OUT/'evidence_summary.csv');latest=evidence[(evidence.run=='pdz2018_portfolio_v1')&(evidence.scheme=='site')].copy()
    coverage=pd.read_csv(OUT/'task_evidence.csv');available=coverage[coverage.count_available]
    gap=json.loads((OUT/'source_refresh.json').read_text())
    result=dict(scope='Evidence rescore and frozen zero-shot static signal check; no adaptive improvement claim',
        evidence=dict(total_tasks=207,count_available_tasks=len(available),count_missing_tasks=int((~coverage.count_available).sum()),
            point_reachable=int(coverage.point_reachable.sum()),count_supported_reachable=int(available.count_reachable.sum()),full_count_supported_reachable=int(available.full_count_reachable.sum()),
            latest_methods=latest.to_dict('records'),biological_intervals_available=False,missing_materials=gap['unresolved']),
        signal=dict(selected_score=chosen,association_gate=association,ready_for_search_gate=all(checks.values()),gate_checks=checks,summary=summary.replace({np.nan:None}).to_dict('records'),
            interpretations=['Evolutionary scores are not measured affinity or kcal/mol','Static initial8+ranked24 coverage is not adaptive search performance','Previously inspected development labels; subgroup results are diagnostic',
                            'No supervised fitting; per-domain reporting is consistency assessment, not leave-domain-out training validation']),
        new_adaptive_queries=0,new_supervised_fits=0,new_agnes_calls=0,sealed_domains_loaded=False,
        next='Discuss evidence and signal results before modifying the search method')
    dump(OUT/'result.json',result)
    # Preserve a targeted diagnostic of the previously identified16 misses, not a new selected primary cohort.
    miss=pd.read_csv(OUT/'missed_rescue_evidence.csv');ranks=pd.read_csv(OUT/'signal_rankings.csv.gz')
    joined=miss.merge(ranks[['task_id','method','rescue','remaining_rank']],on=['task_id','rescue'],how='left');joined.to_csv(OUT/'missed_rescue_rank_check.csv',index=False)
    show=summary.copy();show['method']=show.method.map(LABELS)
    decision='达到预设的进入搜索开发条件；本次仍只做信号检查。' if all(checks.values()) else '未达到预设的进入搜索开发条件。'
    association_text='补偿评分与有利残差有一致方向的描述性相关。' if association else '补偿评分尚未显示跨域一致的有利残差信号。'
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>救援证据与背景条件信号检查</title>
<style>body{{font:16px/1.65 system-ui;max-width:1350px;margin:40px auto;padding:0 24px;color:#18392d;background:#f6faf7}}h2{{font-size:22px;margin-top:32px}}table{{border-collapse:collapse;background:white;width:100%;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #dde6df;text-align:left;overflow-wrap:anywhere}}th{{background:#e7f1e9}}.card{{background:#e6f2e9;padding:20px;border-left:5px solid #347b58}}.scroll{{overflow-x:auto}}</style>
<h1>两项检查：救援证据与背景条件评分</h1><p>全部207个开发任务 · 现有轨迹重评分 · ESM-2 35M固定版本 · 封存测试未读</p>
<div class="card"><b>证据检查</b>：140任务可计算条件计数界，67个PDZ3任务保持缺失；计数支持可达任务{int(available.count_reachable.sum())}个，损伤也得到计数确认的可达任务{int(available.full_count_reachable.sum())}个。<br><b>信号检查</b>：{association_text}{decision}</div>
<h2>已有方法的证据核对</h2><p>表中点命中主分母207；point_success_available与计数支持使用相同140任务。计数界沿用固定校准和39504设计族的同时Poisson界，不等于生物学或总测量置信区间。缺失没有记为阴性。</p>
<div class="scroll">{latest[['method','point_success','available_tasks','point_success_available','count_supported','full_count_supported','missing_count_tasks']].to_html(index=False,border=0)}</div>
<h2>背景条件评分</h2><p>对同一候选位点分别在WT背景、损伤背景下遮蔽该残基并计算替换log-odds；相减得到背景依赖变化，再与反方向变化平均。共4131个公开序列输入，无标签参与推理、无监督拟合、无权重调参。所有方法先保留同样8个WT候选，再一次性排序剩余候选。potential32表示这8个加上前24名的静态覆盖，不能称为已经完成的主动搜索成绩。</p>
<div class="scroll">{show.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}')}</div><h2>分域一致性</h2>{domain.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}')}
<h2>数据缺口</h2><p>本次刷新Dryad和eLife公开目录。Dryad当前列出{gap.get('dryad_total')}个文件：{', '.join(gap.get('dryad_files',[]))}。逐突变重复、PDZ3能量聚合映射、校准协方差和完整实验构建体序列仍未补齐。结论仅限已检查来源。已有作者请求草稿保留，未发送。</p>
<h2>独立验证</h2><p>从源计数重建{v['double_bounds_rebuilt']}个双突变和{v['single_bounds_rebuilt']}个单突变区间；重算{v['saved_trajectories_rescored']}条已保存轨迹；独立CPU复算9个遮蔽上下文及3个双向补偿分数；逐任务复核排名、覆盖与缺失处理。旧文件哈希不变。</p>
<p>该检查为开发探索。只有三个域与九个位点，任务还共享组合基因型；分组区间为描述性敏感性分析。PDB构建体参考上下文不等同已确认的实验全序列，零样本语言模型分数不等于功能测量。</p>
<p><a href="protocol.json">固定协议</a> · <a href="result.json">全部结果</a> · <a href="verification.json">验证</a> · <a href="source_refresh.json">公开来源检查</a> · <a href="missed_rescue_rank_check.csv">16个漏检任务诊断</a></p></html>'''
    (OUT/'report.html').write_text(body)
    dump(OUT/'final_artifact_hashes.json',hashes([p for p in OUT.rglob('*') if p.is_file() and p.name!='final_artifact_hashes.json' and p.suffix!='.log']))
    print(latest[['method','point_success_available','count_supported','full_count_supported']].to_string(index=False));print(summary.to_string(index=False));print(json.dumps(result['signal']['gate_checks'],indent=2))


if __name__=='__main__':main()
