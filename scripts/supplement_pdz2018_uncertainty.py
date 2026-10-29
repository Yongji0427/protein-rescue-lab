"""Versioned post-run scoring supplement; never change previous trials."""
from datetime import datetime,timezone
import hashlib
import html
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from audit_pdz2018 import fit_calibration
from pdz2018_data import ROOT,DOMAINS,RT,AA,dump,hashes
from pdz2018_uncertainty import DESIGNED_FAMILY,FAMILY_ALPHA,TAIL_ALPHA,score_counts,missing_pdz3

OUT=ROOT/'experiments/pdz2018_uncertainty_v1'
PILOT=ROOT/'experiments/pdz2018_point_dev_v1'


def main():
    assert not (OUT/'freeze.json').exists(),'Never overwrite a completed scoring supplement'
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(version=OUT.name,status='post-run conditional counting sensitivity; not formal reliable-rescue endpoint',
        source_pilot=PILOT.name,loaded_domains=list(DOMAINS),unread_test_domains=['shank3','zo1'],
        confidence_scope='simultaneous count-ratio bounds across fixed designed development family, conditional on frozen calibration and Poisson-count model',
        designed_family=DESIGNED_FAMILY,family_alpha=FAMILY_ALPHA,each_tail_alpha=TAIL_ALPHA,
        method='two-sided Clopper-Pearson conditional Poisson rate-ratio; Bonferroni both tails over all designed variants including WT; mutant lower/WT upper and mutant upper/WT lower',
        calibration='reuse pre-existing public-single-derived alpha/fraction WT; no fit to new double outcomes',
        source_mapping_gate='unchanged original RMSE <= .001 kcal for singles and finite doubles; PDZ3 unavailable',
        primary_point_endpoint='unchanged: published double deltaG-derived affinity >=.5WT',
        supplementary_combo_support='published point>=.5 AND simultaneous counting lower>=.5',
        supplementary_full_task_support='combo support AND damaging-single simultaneous upper<=.2',
        harmful_R='optional additional public-single simultaneous upper<1; no exclusions or method changes',
        unsupported='biological repeat variance, fitted-calibration parameter uncertainty, and PDZ3 replicate alignment',
        search_trajectories='frozen original trials only; no new query or model call',
        selection_boundary='fixed design alpha independent of observed count/energy quality filters; all original tasks retained; unavailable flags never converted to negatives')
    dump(OUT/'protocol.json',protocol)
    inputs=[ROOT/'scripts/pdz2018_uncertainty.py',ROOT/'scripts/supplement_pdz2018_uncertainty.py',OUT/'protocol.json',
        ROOT/'data/pdz2018/source/Salinas_Ranganathan_data.mat',ROOT/'experiments/dataset_feasibility_v1/pdz2018_calibration_QC.csv',
        PILOT/'protocol.json',PILOT/'fixed_runs.csv',PILOT/'fixed_query_log.csv.gz',PILOT/'verification.json']
    inputs+=list((PILOT/'queried_results').glob('*.json'))
    dump(OUT/'freeze.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes(inputs),
        evidence='development post-run uncertainty supplement, not new formal validation'))
    source=loadmat(ROOT/'data/pdz2018/source/Salinas_Ranganathan_data.mat',variable_names=list(DOMAINS),squeeze_me=True,struct_as_record=False)
    assert 'shank3' not in source and 'zo1' not in source
    cal=pd.read_csv(ROOT/'experiments/dataset_feasibility_v1/pdz2018_calibration_QC.csv').set_index('domain')
    single_bounds={};combo_bounds={};single_records=[];combo_records=[];qc=[]
    for domain,(seq,_) in DOMAINS.items():
        o=source[domain];available=bool(cal.loc[domain,'count_support_available'])
        alpha,f=cal.loc[domain,['alpha','fraction_WT']]
        fitted,error,energy=fit_calibration(o)
        assert np.allclose(fitted,[alpha,f],rtol=1e-9)
        x=np.log(o.selected_double_counts/o.unselected_double_counts/(o.selected_wt_counts/o.unselected_wt_counts))
        mask=np.isfinite(x)&np.isfinite(o.deltaG_double)
        residual=energy(x[mask],fitted)-o.deltaG_double[mask]
        double_rmse=float(np.sqrt(np.mean(residual**2)))
        assert available==(error<=.001 and double_rmse<=.001)
        qc.append(dict(domain=domain,count_interval_available=available,single_rmse=error,double_rmse=double_rmse,
            max_absolute_double_mapping_discrepancy=float(max(abs(residual))),biological_interval_available=False,
            reason='conditional counting only, calibration fixed' if available else missing_pdz3()['reason']))
        for i in range(9):
            for a in range(20):
                g=o.deltaG_single[a,i]
                if AA[a]==seq[i] or not np.isfinite(g):continue
                key=f'{seq[i]}{i+1}{AA[a]}'
                low=high=None
                if available:
                    low,high=score_counts(o.selected_single_counts[a,i],o.unselected_single_counts[a,i],o.selected_wt_counts,o.unselected_wt_counts,alpha,f)
                    low,high=float(low),float(high)
                record=dict(domain=domain,mutation=key,point_affinity=float(np.exp(-g/RT)),count_available=available,
                    counting_lower=low,counting_upper=high,damage_confirmed=high<=.2 if available else None,
                    harmful_confirmed=high<1 if available else None)
                single_bounds[domain,key]=record;single_records.append(record)
        coordinates=[]
        for i in range(9):
            for j in range(i+1,9):
                for a in range(20):
                    for b in range(20):
                        d,r=f'{seq[i]}{i+1}{AA[a]}',f'{seq[j]}{j+1}{AA[b]}'
                        g,n,s=o.deltaG_double[i,j,a,b],o.unselected_double_counts[i,j,a,b],o.selected_double_counts[i,j,a,b]
                        if (domain,d) not in single_bounds or (domain,r) not in single_bounds or not np.isfinite(g) or not np.isfinite(n) or n<=0 or not np.isfinite(s):continue
                        coordinates.append((d,r,float(g),float(s),float(n)))
        table=pd.DataFrame(coordinates,columns=['d','r','g','s','n'])
        if available:
            lower,upper=score_counts(table.s.to_numpy(),table.n.to_numpy(),o.selected_wt_counts,o.unselected_wt_counts,alpha,f)
        else:lower=upper=[None]*len(table)
        for row,low,high in zip(table.itertuples(),lower,upper):
            point=float(np.exp(-row.g/RT));record=dict(domain=domain,mutation1=row.d,mutation2=row.r,point_affinity=point,
                count_available=available,counting_lower=float(low) if available else None,counting_upper=float(high) if available else None,
                combo_supported=bool(point>=.5 and low>=.5) if available else None)
            combo_records.append(record);combo_bounds[domain,row.d,row.r]=record;combo_bounds[domain,row.r,row.d]=record
    pd.DataFrame(qc).to_csv(OUT/'calibration_QC.csv',index=False)
    pd.DataFrame(single_records).to_csv(OUT/'single_bounds.csv',index=False)
    pd.DataFrame(combo_records).to_csv(OUT/'double_bounds.csv',index=False)
    manifest=pd.read_csv(PILOT/'task_manifest.csv');original=pd.read_csv(PILOT/'fixed_runs.csv')
    scored=[]
    for task_id,method,repeat in original[['task_id','method','repeat']].itertuples(index=False,name=None):
        domain,d=task_id.split('__');s=single_bounds[domain,d]
        scored.append(dict(task_id=task_id,domain=domain,method=method,repeat=repeat,
            count_available=s['count_available'],damage_confirmed=s['damage_confirmed'],combo_supported=False if s['count_available'] else None,
            full_task_supported=False if s['count_available'] else None,harmful_R_combo_supported=False if s['count_available'] else None))
    lookup={(r['task_id'],r['method'],r['repeat']):r for r in scored}
    for chunk in pd.read_csv(PILOT/'fixed_query_log.csv.gz',chunksize=100000):
        for (task_id,method,repeat),group in chunk.groupby(['task_id','method','repeat'],sort=False):
            dest=lookup[task_id,method,int(repeat)];domain,d=task_id.split('__')
            if not dest['count_available']:continue
            supported=[r for r in group.rescue if combo_bounds[domain,d,r]['combo_supported']]
            dest['combo_supported']|=bool(supported)
            dest['full_task_supported']|=bool(supported and dest['damage_confirmed'])
            dest['harmful_R_combo_supported']|=any(single_bounds[domain,r]['harmful_confirmed'] for r in supported)
    frame=pd.DataFrame(scored);frame.to_csv(OUT/'fixed_rescored_runs.csv',index=False)
    core=[];p=json.loads((PILOT/'protocol.json').read_text())
    for task_id in p['damage_tasks']:
        for method in p['methods']:core.append(lookup[task_id,method,0].copy())
        s=json.loads((PILOT/'queried_results'/(task_id+'.json')).read_text());domain,d=task_id.split('__');single=single_bounds[domain,d];available=single['count_available']
        combo=any(combo_bounds[domain,d,r['rescue']]['combo_supported'] for r in s['queries']) if available else None
        core.append(dict(task_id=task_id,domain=domain,method='Agnes_AGH',repeat=0,count_available=available,
            damage_confirmed=single['damage_confirmed'],combo_supported=combo,
            full_task_supported=bool(combo and single['damage_confirmed']) if available else None,
            harmful_R_combo_supported=any(combo_bounds[domain,d,r['rescue']]['combo_supported'] and single_bounds[domain,r['rescue']]['harmful_confirmed'] for r in s['queries']) if available else None))
    core=pd.DataFrame(core);core.to_csv(OUT/'core_rescored.csv',index=False)
    full=frame[frame.count_available].groupby(['task_id','method'])[['combo_supported','full_task_supported','harmful_R_combo_supported']].mean().groupby('method').agg(['sum','mean'])
    full.to_csv(OUT/'full_comparison.csv')
    core_summary=core.groupby('method').agg(planned_tasks=('task_id','size'),available_tasks=('count_available','sum'),
        combo_supported=('combo_supported','sum'),full_task_supported=('full_task_supported','sum'),harmful_R_combo_supported=('harmful_R_combo_supported','sum')).reset_index()
    core_summary.to_csv(OUT/'core_comparison.csv',index=False)
    coverage=[]
    for t in manifest.itertuples():
        available=single_bounds[t.domain,t.damage]['count_available']
        pairs=[r for r in combo_records if r['domain']==t.domain and t.damage in [r['mutation1'],r['mutation2']]]
        coverage.append(dict(task_id=t.task_id,domain=t.domain,count_available=available,
            damage_confirmed=single_bounds[t.domain,t.damage]['damage_confirmed'],
            reachable_supported=any(r['combo_supported'] for r in pairs) if available else None))
    coverage=pd.DataFrame(coverage);coverage.to_csv(OUT/'task_coverage.csv',index=False)
    decision=dict(status='two-domain conditional count endpoint completed; full reliability blocked',
        total_tasks=len(manifest),count_available_tasks=int(coverage.count_available.sum()),unavailable_tasks=int((~coverage.count_available).sum()),
        biological_CI_available_tasks=0,PDZ3='unavailable',new_model_calls=0,new_label_queries=0,test_arrays_loaded=False,
        full_reliability_requirements=['per-replicate counts/energies and genotype mapping','standard-curve parameters, covariance or raw repeated calibration measurements','energy aggregation and quality/saturation masks'],
        point_results_changed=False,calibration_parameters_refit=False,
        note='new conditional simultaneous bounds do not resolve PDZ3 or biological/calibration uncertainty; counts alone cannot fill that gap')
    dump(OUT/'decision.json',decision)
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>PDZ 不确定性补充</title><style>body{{font:16px/1.7 system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#173044}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{border-bottom:1px solid #ddd;padding:8px;text-align:left}}.note{{background:#eef4f6;padding:18px}}</style>
<h1>PDZ2018 不确定性补充</h1><p class="note">已补齐两个可校准结构域的条件计数界、WT 计数不确定性和搜索多重比较校正。140 个开发任务可提供该补充端点；67 个 PDZ3 任务保持 unavailable。逐突变生物学误差和标准曲线参数不确定性未取得，因此完整可靠救援端点仍未补齐。</p>
<h2>评分规则</h2><p>固定设计族为 3 × (171 单突变 + 12996 双突变 + 1 WT) = {DESIGNED_FAMILY:,}，逐尾概率 {TAIL_ALPHA:.9g}。条件 Poisson 率比使用 Clopper–Pearson 区间，Bonferroni 覆盖两尾及全部设计基因型；组合下界除以 WT 上界，上界除以 WT 下界。固定整库设计防止自适应选择、只看已查询组合或质量筛选缩小校正范围。</p>
<p>计数稳健组合要求原点值 ≥ 0.5 WT 且计数下界 ≥ 0.5。完整计数稳健任务再要求损伤单突变上界 ≤ 0.2。R 上界 &lt; 1 作为另一个条件单独报告。不改选任务、不改搜索、不新增模型调用；67 个未知状态不会记成失败。</p>
<h2>相同 32 个任务的事后重评分</h2>{core_summary.to_html(index=False,border=0)}<p>每种方法 32 个任务，其中 21 个可提供计数界，11 个 PDZ3 缺失；该 21 任务比较仅是可用子集的描述，原主要分母与点值结果不变。</p>
<h2>全量可用子集：140 个任务</h2>{full.to_html(border=0)}<p>WT 为确定性对照，结构策略为 100 个搜索种子的平均成功任务数。种子不代表生物学重复。</p>
<h2>校准核对</h2>{pd.DataFrame(qc).to_html(index=False,border=0)}<p>此计数界对固定校准给出的计数隐含亲和力成立，不是对作者公布能量的总误差保证。上表披露其逐双突变映射差异；原 RMSE 门控未改变，也没有用新双突变标签重新拟合校准。</p>
<h2>公开数据追索结果</h2><p>出版社元数据、Dryad 文件目录、MAT 结构、透明报告表及作者网站没有找到逐重复数值附件。报告表指向同一 MATLAB workspace；图 1 supplement 2 报告四次 PDZ3 实验，但没有逐突变误差表。不能把 0.3 kcal/mol 平均误差、0.06 位置对平均误差或图中的离散程度直接复制为所有突变的标准误。</p>
<p><a href="data_request_draft.txt">所需数据与作者请求草稿（未发送）</a> · <a href="decision.json">当前可用性</a> · <a href="protocol.json">补充协议</a> · <a href="verification.json">核验</a> · <a href="https://doi.org/10.7554/eLife.34300">原文</a> · <a href="https://datadryad.org/dataset/doi%3A10.5061/dryad.gk4m1">原始数据</a></p></html>'''
    (OUT/'report.html').write_text(body)
    print(core_summary.to_string(index=False));print(full.to_string());print(json.dumps(decision,indent=2))


if __name__=='__main__':main()
