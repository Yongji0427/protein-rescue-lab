"""Render matched-cohort Agnes/AGH results with execution failures disclosed."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/gb1_agnes_v1'


def main():
    summary = pd.read_csv(OUT/'comparison.csv')
    audit = json.loads((OUT/'verification.json').read_text())
    protocol = json.loads((OUT/'protocol.json').read_text())
    corrections = json.loads((OUT/'completion_corrections.json').read_text())
    first_incomplete, terminal_noncompleted = [], []
    for damage in protocol['damage_tasks']:
        first = json.loads((OUT/'first_turn_results'/f'{damage}.json').read_text())
        trial = json.loads((OUT/'agh_events'/f'{damage}.run.json').read_text())
        if len(first['queries'])<32:first_incomplete.append(damage)
        if trial.get('outcome',{}).get('reason')!='completed':terminal_noncompleted.append({'damage':damage,'outcome':trial.get('outcome',trial.get('error'))})
    audit['first_turn_budget_incomplete_tasks'] = first_incomplete
    audit['first_turn_terminal_noncompleted'] = terminal_noncompleted
    (OUT/'execution_summary.json').write_text(json.dumps(audit,indent=2)+'\n')
    names={'random':'随机搜索（100 次）','wt_single_rank':'WT 单突变排序','direct':'直接反馈规则（seed 0）','contrast':'对照驱动规则（seed 0）','agnes_AGH':'Agnes＋官方 AGH（单次）'}
    english={'random':'Random','wt_single_rank':'WT ranking','direct':'Direct rule','contrast':'Contrast rule','agnes_AGH':'Agnes + AGH'}
    colors={'random':'#999da3','wt_single_rank':'#376d9b','direct':'#927fa6','contrast':'#bd783f','agnes_AGH':'#467b62'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for method in names:
        data=summary.loc[summary.method==method]
        for ax,col in zip(axes,['success_rate','count_robust_success_rate']):
            ax.plot(data.budget,data[col]*100,'o-',label=english[method],color=colors[method],lw=1.5,ms=4)
    axes[0].axhline(audit['reachability_tasks']/32*100,color='#b7b7b7',ls='--')
    axes[0].legend(loc='upper right',fontsize=8,frameon=False)
    axes[0].text(8, audit['reachability_tasks']/32*100+1, f"Pool ceiling: {audit['reachability_tasks']}/32", fontsize=8, color='#737373')
    axes[0].text(17, 29.3, 'WT / rules / Agnes: 9/32', fontsize=8, color='#467b62')
    for ax in axes:ax.set(xlabel='New combination-label queries',ylabel='Success (%)',xticks=[8,16,32],ylim=(0,50))
    axes[0].set_title('A  Rescue at 50% WT')
    axes[1].set_title('B  Count-interval-confirmed rescue')
    fig.suptitle('32 matched development tasks | official AGH | one Agnes run/task',fontsize=12)
    fig.savefig(OUT/'agnes_results.png',dpi=180);fig.savefig(OUT/'agnes_results.svg');plt.close(fig)
    table=summary.copy();table.method=table.method.map(names)
    for col in ['success_rate','count_robust_success_rate']:table[col]=table[col].map(lambda x:f'{100*x:.2f}%')
    table.mean_best_W=table.mean_best_W.map(lambda x:f'{x:.4f}')
    table.columns=['方法','预算','救援率','计数区间确认救援率','最佳结合均值 / WT','任务数']
    model=summary.query("method=='agnes_AGH' and budget==32").iloc[0]
    wt=summary.query("method=='wt_single_rank' and budget==32").iloc[0]
    rule=summary.query("method=='contrast' and budget==32").iloc[0]
    best=[]
    for damage in protocol['damage_tasks']:
        state=json.loads((OUT/'queried_results'/f'{damage}.json').read_text())
        winner=max(state['queries'],key=lambda x:x['W_display'])
        if winner['count_ci_low']>=.5:
            best.append({'damage':damage,'rescue':winner['rescue'],'combination_W':winner['W_display'],
                'combination_lower':winner['count_ci_low'],'WT_rescue_W':math.exp(winner['WT_rescue_lnW']),
                'damage_W_display':math.exp(state['damage_lnW'])})
    pd.DataFrame(best).to_csv(OUT/'confirmed_examples.csv',index=False)
    parts=[
        '<h1>Agnes＋官方 AGH：初步搜索实验</h1><p class="tag">32 个固定选取的开发任务 · 每任务一次模型运行 · 公开实验回放 · 预留评测位置未使用</p>',
        f'<p><b>真实模型决策结果已获得。</b>32 次查询下，Agnes＋AGH 的救援率为 {100*model.success_rate:.2f}%（{round(model.success_rate*32)}/32）；同一任务集合的 WT 排序为 {100*wt.success_rate:.2f}%，对照驱动固定规则为 {100*rule.success_rate:.2f}%。只比较本轮同一 32 个任务，不能与之前 188 个任务的总体数字直接比较。</p>',
        '<img src="agnes_results.png" alt="Agnes matched-cohort results">',
        '<h2>实验条件</h2><p>使用中国站官方接口 https://api.agnes-ai.cn/v1 与 agnes-3.0-flash，经官方 Agnes Harness 源码构建的运行时执行。任务取开发集合中 SHA256(gb1_agnes_v1|damage) 最小的 32 个背景，与救援结果无关。共同起始 8 次 WT 排序查询计入预算，之后每次通过 query_rescue_batch 提交四个候选。</p>',
        '<p>候选菜单仅依据可见的单突变和已查询反馈生成，包括 WT 排序、对照估计、未探索位置和固定规则提案；模型必须保留一项 WT 锚定候选，可选择其他三项。标签由独立 evaluator 进程提供，工具不提供文件读取、shell 或自由查看数据库能力。AGH 会话工作目录与标签文件分离；实际记录中的工具调用均为该受限查询工具。</p>',
        '<p>这不是完整 188 个任务评测，也不是正式预留集结果。模型每任务只运行一次，未测量模型随机性。固定规则比较采用同一任务集合的 seed 0，随机搜索采用各任务的 100 次开发重复。</p>',
        '<h2>同一任务集合的全部比较</h2>',table.to_html(index=False),
        f'<p>当前候选池内可达背景为 {audit["reachability_tasks"]}/32。原本不可达或执行异常的任务均保留在分母中。计数区间沿用先前固定归一化下的 Poisson 计数比敏感性检查，不代表新湿实验或生物学重复。</p>',
        '<h2>执行异常与核验反馈</h2>',
        f'<p>首个用户任务调用中，有 {len(first_incomplete)} 个任务未达到 32 次：{", ".join(first_incomplete) or "无"}。模型输出了工具样式的普通文本，该文本没有执行且没有揭示新标签。根据记录的统一完成度规则，对所有预算不足任务在原会话中追加最多两次相同核验反馈；保持同一 oracle、同一损伤和同一总预算，不重启搜索或增加标签预算。原始结果、核验反馈和最终结果均保存。</p>',
        pd.DataFrame(corrections).to_html(index=False) if corrections else '<p>无需额外完成度反馈。</p>',
        f'<p>查询预算最终完成 {audit["complete_tasks"]}/32。另有首轮非 completed 终止状态：{json.dumps(terminal_noncompleted,ensure_ascii=False)}。这些状态与“是否已查询到功能恢复记录”分别统计；不能因为查询结果存在而声称整个执行流程无异常。</p>',
        '<h2>证据、开销与边界</h2>',
        f'<p>已核验 {audit["point_queries"]:,} 条组合标签记录、{audit["model_selected_batches"]} 个模型提交批次、{audit["unique_AGH_sessions"]} 个独立 AGH 会话。菜单成员、损伤保留、异位点单替换、预算上限和模型身份均检查通过。预留位置组合评测数为 0。API key 在实验输出中的匹配数为 0。</p>',
        f'<p>会话 cost/ledger 记录的推理调用为 {audit["ledger_inference_calls"]} 次，输入 token {audit["tokens"]["input"]:,}，输出 token {audit["tokens"]["output"]:,}，缓存读取 token {audit["tokens"]["cacheRead"]:,}。这里只报告账本计数，不将它等同于实验查询次数，也不将估计费用当作实际平台账单。</p>',
        '<p>本轮增加的完成度核验是在发现执行问题后补入的开发工程调整；所有版本和失败记录保留，不能据此主张独立冻结的正式改进。模型可能具有预训练中的 GB1 知识，工具标签隔离不能排除这一点。决策理由是搜索记录，不是机制证据。</p>',
        f'<p>官方 AGH 提交：{audit["official_AGH_commit"]}；实验协议 SHA256：{audit["protocol_sha256"]}。</p>',
        '<p>数据与计数说明见 <a href="../gb1_pilot_v1/report.html">基线报告</a>；固定规则方法见 <a href="../gb1_adaptive_v1/report.html">自适应规则报告</a>。本轮真实 AGH 事件记录位于 agh_events，预算查询结果位于 queried_results。</p>',
    ]
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Agnes 初步实验</title><style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#243343;line-height:1.7}.tag{color:#687789}img{width:100%}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}td,th{padding:9px 12px;border-bottom:1px solid #dce4eb;text-align:left}th{background:#edf2f7}h2{margin-top:30px}</style><body>'+'\n'.join(parts)+'</body></html>'
    (OUT/'report.html').write_text(page)
    print('Agnes32:',model.to_dict());print('Matched WT32:',wt.to_dict());print('Matched contrast32:',rule.to_dict())
    print('Report:',OUT/'report.html')


if __name__=='__main__':main()
