"""Report factorial attribution, real-model integration and one frozen holdout."""
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment_data import ROOT,load_frames

OUT=ROOT/'experiments/gb1_three_steps_v1'


def table(frame):return frame.to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}')


def paired_bootstrap(frame,method,reference,metric):
    pivot=frame.pivot(index='damage',columns='method',values=metric)
    delta=(pivot[method]-pivot[reference]).astype(float)
    pos=delta.index.str[1:-1].astype(int);groups=sorted(set(pos));sums=np.array([delta[pos==g].sum() for g in groups]);counts=np.array([(pos==g).sum() for g in groups])
    rng=np.random.default_rng(20261008);draw=rng.integers(0,len(groups),(10000,len(groups)))
    distribution=sums[draw].sum(1)/counts[draw].sum(1)
    return dict(method=method,reference=reference,metric=metric,delta_pp=100*delta.mean(),lower_pp=100*np.quantile(distribution,.025),upper_pp=100*np.quantile(distribution,.975),clusters=len(groups),gained_tasks=int((delta>0).sum()),lost_tasks=int((delta<0).sum()))


def main():
    OUT.mkdir(exist_ok=True)
    factor=ROOT/'experiments/gb1_factor_v2';dev=ROOT/'experiments/gb1_agnes_structure_dev_v1';final=ROOT/'experiments/gb1_final_holdout_v1'
    for folder in [factor,dev,final]:assert json.loads((folder/'verification.json').read_text())['status']=='passed'
    f=pd.read_csv(factor/'summary.csv');d=pd.read_csv(dev/'comparison.csv');h=pd.read_csv(final/'comparison.csv')
    wt=pd.read_csv(ROOT/'experiments/gb1_pilot_v1/summary.csv').query("method=='wt_single_rank'")
    combined=pd.concat([f,wt],ignore_index=True);combined[combined.budget==32].to_csv(OUT/'development_factor.csv',index=False)
    d[d.budget==32].to_csv(OUT/'agnes_development.csv',index=False);h[h.budget==32].to_csv(OUT/'holdout_comparison.csv',index=False)
    p=json.loads((final/'protocol.json').read_text());singles,frames=load_frames(p,final)
    reach=pd.DataFrame([dict(damage=damage,candidates=len(frame),point_reachable=bool((frame.W_display>=.5).any()),count_robust_reachable=bool((frame.ci_low>=.5).any()),best_available_W=float(frame.W_display.max())) for damage,frame in frames.items()])
    reach.to_csv(final/'reachability_after_evaluation.csv',index=False)
    run=pd.read_csv(final/'runs.csv');fixed=pd.read_csv(final/'fixed_policies/runs.csv')
    fixed=fixed[(fixed['repeat']==0)&(fixed.budget==32)].copy();fixed['method']=fixed.method.map({'wt_rank':'WT_rank',p['selected_method']:'fixed_full_pool'})
    all32=pd.concat([run[run.budget==32],fixed],ignore_index=True)
    pairs=all32.pivot(index='damage',columns='method',values=['success','count_robust_success','best_W'])
    pairs.columns=['_'.join(x) for x in pairs.columns];pairs.reset_index().to_csv(OUT/'heldout_task_table.csv',index=False)
    stats=[]
    for method in ['fixed_full_pool','fixed_same_menu','Agnes_structure']:
        for ref in ['WT_rank']+(['fixed_same_menu'] if method=='Agnes_structure' else []):
            for metric in ['success','count_robust_success']:stats.append(paired_bootstrap(all32,method,ref,metric))
    stats=pd.DataFrame(stats);stats.to_csv(OUT/'position_bootstrap.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
    for ax,frame,title in zip(axes,[combined,d,h],['Development factorial (188)','AGH development core (32)','Frozen holdout (51)']):
        for method,g in frame.groupby('method'):
            label={'wt_single_rank':'WT rank','no_feedback_contact':'WT + contact','no_feedback_gate_random':'WT + gated random','feedback_contact':'Feedback + contact','feedback_gate_random':'Feedback + gated random','Agnes_structure':'Agnes + structure','Agnes_previous_menu':'Agnes previous','fixed_same_menu':'Fixed same menu','WT_rank':'WT rank','fixed_full_pool':'Fixed full pool'}.get(method,method)
            ax.plot(g.budget,g.success_rate*100,marker='o',label=label)
        ax.set(xlabel='Unique queries',ylabel='Point rescue success (%)',title=title,xticks=[8,16,24,32]);ax.grid(alpha=.2);ax.legend(fontsize=6,frameon=False)
    fig.savefig(OUT/'three_steps.png',dpi=180);fig.savefig(OUT/'three_steps.svg');plt.close(fig)
    dev_a=json.loads((dev/'verification.json').read_text());final_a=json.loads((final/'verification.json').read_text())
    frozen=json.loads((final/'freeze.json').read_text())
    result=dict(factorial_selected=p['selected_method'],development_tasks=188,Agnes_development_core_tasks=32,debug_tasks=2,heldout_tasks=len(reach),heldout_damage_positions=sorted({int(x[1:-1]) for x in reach.damage}),heldout_point_reachable=int(reach.point_reachable.sum()),heldout_count_robust_reachable=int(reach.count_robust_reachable.sum()),development_AGH=dev_a,heldout_AGH=final_a,freeze_utc=frozen['frozen_at_utc'],evidence='single frozen retrospective holdout, no new wet-lab experiments; historical/pretrained knowledge cannot be excluded; development selection does not establish heldout improvement')
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    primary=stats[(stats.method=='Agnes_structure')&(stats.reference=='WT_rank')&(stats.metric=='success')].iloc[0]
    pages='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>蛋白救援实验室：三步完整结果</title><style>body{max-width:1250px;margin:35px auto;padding:0 25px;font:16px/1.75 system-ui;color:#23384b}h1,h2{color:#143a55}table{border-collapse:collapse;font-size:13px}td,th{padding:7px 10px;border-bottom:1px solid #ddd}img{width:100%}.note{padding:15px;background:#fff1d9}a{color:#08688b}</style>
<h1>蛋白救援实验室：归因 → Agnes 集成 → 冻结验证</h1>
<p class="note">三个阶段使用不同队列，不能将各阶段成功率直接比较。主指标始终是组合点值 W≥0.5；次指标是固定 WT 归一化的 90% 条件计数区间下界≥0.5，不能代替生物重复验证。Agnes 每任务一次模型试验，固定策略100种子仅作补充，最终主表使用同一seed0。</p>
<h2>第一步：开发集归因</h2><p>188个背景；2×2 对照共享相同中性替换和原生疏水残基条件。结构组使用真实距离，随机组在相同资格的候选中均匀探索。关闭反馈时两个排序名额直接按WT先验填充。结果支持将最终固定策略简化为 no_feedback_contact；没有参数扫描。</p>'''+table(combined[combined.budget==32])+'''
<p>该版本保留 F256D 的原有成功，并补到 V265G/V265W。效应来自开发集，不能视为独立验证；选择规则在归因实验运行前写入协议。</p>
<h2>第二步：真实 Agnes 结构菜单</h2><p>32个原身份哈希队列作为核心比较；两个已知V265背景仅作事后功能调试，不混入核心成功率。固定策略与Agnes使用相同公共菜单生成器及先验，各自只看到自己的已查反馈。上一版Agnes仅作描述性参照。</p>'''+table(d[d.budget==32])+'''<p>调试案例：</p>'''+table(pd.read_csv(dev/'debug_runs.csv').query('budget==32'))+'''
<h2>第三步：一次冻结封存验证</h2><p>51个公开单突变定义的损伤背景，位于7个封存位置。候选救援限开发位置；包括原本不可救援的背景。所有方法共享候选池和前8次查询。代码、协议、提示词、公开参考数据、原始数据、AGH核心文件和插件均在读封存组合前做哈希冻结。评测后才计算可达上限。</p>'''+table(h[h.budget==32])+f'''<p>封存池主指标可达背景 {result['heldout_point_reachable']}/51；计数下界达标可达背景 {result['heldout_count_robust_reachable']}/51。不能因没有提升而重新挑选任务或调整本次方法。</p>
<p>Agnes相对WT的主指标差值 {primary.delta_pp:.2f} 个百分点。95%位置分组bootstrap区间 [{primary.lower_pp:.2f}, {primary.upper_pp:.2f}]；只有7个位置，属于有限的任务相关性敏感性分析。</p>'''+table(stats)+'''
<img src="three_steps.png" alt="three stages with matched comparison curves">
<h2>执行与验证</h2><p>开发AGH完成 '''+str(dev_a['complete_tasks'])+'/'+str(dev_a['tasks'])+'''，封存AGH完成 '''+str(final_a['complete_tasks'])+'/'+str(final_a['tasks'])+'''。原始工具错误、预算补全和终止状态均保留；没有按结果替换失败任务。各阶段验证通过逐条标签核对、独立菜单重建、预算及首8次匹配、唯一AGH会话、调用模型和工具、密钥泄露扫描。</p>
<p>本项目使用公开历史实验回放，不保证预训练模型未见过该数据，不能将输出解释为新的湿实验或物理机制验证。</p>
<p><a href="result.json">汇总审计</a> · <a href="heldout_task_table.csv">封存逐任务表</a> · <a href="position_bootstrap.csv">成对统计</a> · <a href="../gb1_final_holdout_v1/freeze.json">冻结记录</a> · <a href="../gb1_final_holdout_v1/verification.json">AGH验证</a></p></html>'''
    (OUT/'report.html').write_text(pages)
    print(json.dumps({k:result[k] for k in ['factorial_selected','heldout_tasks','heldout_point_reachable','heldout_count_robust_reachable']},indent=2));print(h[h.budget==32].to_string(index=False));print(stats.to_string(index=False))


if __name__=='__main__':main()
