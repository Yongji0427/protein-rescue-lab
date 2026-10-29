"""Compare frozen development versions without replacing the primary endpoint."""
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/gb1_structure_v1'


def main():
    new = pd.read_csv(OUT / 'runs.csv')
    old = pd.read_csv(ROOT / 'experiments/gb1_adaptive_v1/runs.csv')
    old = old[old.method == 'contrast'].copy(); old['method'] = 'contrast_v1'
    pilot = pd.read_csv(ROOT / 'experiments/gb1_pilot_v1/runs.csv')
    all_runs = pd.concat([pilot, old, new], ignore_index=True)
    columns = ['success', 'count_robust_success', 'strong_success', 'best_W']
    comparison = all_runs.groupby(['method', 'budget'])[columns].mean().reset_index()
    comparison.to_csv(OUT / 'comparison.csv', index=False)
    per = all_runs[all_runs.budget == 32].groupby(['damage', 'method'])[columns].mean().reset_index()
    wt = per[per.method == 'wt_single_rank'].set_index('damage')
    deltas = per[per.method.isin(['random_full', 'contact', 'packing'])].copy()
    for c in columns:
        deltas[c + '_delta_vs_WT'] = deltas[c] - deltas.damage.map(wt[c])
    deltas.to_csv(OUT / 'task_deltas.csv', index=False)
    changed = deltas[(deltas.success_delta_vs_WT.abs() > 1e-10) |
                     (deltas.count_robust_success_delta_vs_WT.abs() > 1e-10)]
    changed.to_csv(OUT / 'changed_tasks.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    names = ['wt_single_rank', 'contrast_v1', 'random_full', 'contact', 'packing']
    labels = ['WT rank', 'Contrast v1', 'Random full', 'Contact', 'Packing']
    for ax, metric, title in zip(axes, ['success', 'count_robust_success'],
                                ['Primary: point rescue', 'Secondary: count-robust rescue']):
        for method, label in zip(names, labels):
            frame = comparison[comparison.method == method]
            ax.plot(frame.budget, 100 * frame[metric], marker='o', label=label)
        ax.set(xlabel='Unique queries per damage', ylabel='Successful tasks (%)', title=title, xticks=[8,16,24,32])
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('GB1 development replay: 188 tasks, identical first eight queries', fontsize=12)
    fig.savefig(OUT / 'comparison.png', dpi=180)
    fig.savefig(OUT / 'comparison.svg')
    plt.close(fig)
    table = comparison[comparison.budget == 32].copy()
    for c in columns[:3]: table[c] = table[c].map(lambda x: f'{100*x:.4f}%')
    verification = json.loads((OUT / 'verification.json').read_text())
    protocol = json.loads((OUT / 'protocol.json').read_text())
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>蛋白救援：结构探索实验</title>
<style>body{max-width:1150px;margin:35px auto;padding:0 25px;font:16px/1.7 system-ui;color:#213547}h1,h2{color:#12324a}table{border-collapse:collapse;font-size:14px}td,th{padding:8px 12px;border-bottom:1px solid #ddd}img{width:100%}.note{padding:16px;background:#fff4db}a{color:#08658c}</style>
<h1>蛋白救援：结构探索实验 v1</h1>
<p>开发集 188 个损伤背景，每个预算 32、批量 4。新版本各运行 100 个随机种子；WT 排序是确定性对照。所有方法共享前 8 次 WT 查询。</p>
<p class="note">这是查看开发集漏检任务后提出的探索实验。没有独立验证或封存集结果，也没有运行 Agnes v2。100 个重复是搜索随机种子，不是 100 次独立生物实验。</p>
<h2>32 次查询的结果</h2><p>邻近探索得到小幅开发增益：主指标 43.1011%，WT 为 42.5532%；计数稳健指标 38.2979%，WT 为 37.2340%。邻近版本补到 V265G、V265W 的计数稳健救援，但 F256D 的点值成功率由 WT 的 100% 降为 2%。体积互补版本没有改善，不能支持其有效性或机制解释。</p>''' + table.to_html(index=False, border=0) + '''
<img src="comparison.png" alt="primary and secondary rescue success curves">
<h2>这轮改变了什么</h2><p>每批保留一个 WT 锚点和两个原有对比反馈候选，只改变第四个探索候选。全池随机控制允许重复探索同一位点；contact 根据参考结构距离优先；packing 再加入体积增减方向互补。结构分数为零时退回全池随机。</p>
<p>中性替换和原生疏水残基的范围、6 Å 距离尺度、体积指数和排序规则均在运行前写入 protocol.json。没有特定位点或救援答案的硬编码。</p>
<h2>逐背景变化</h2><p>数值为该背景 100 个种子的成功比例，以及相对 WT 排序的差值。</p>''' + changed.to_html(index=False, border=0, float_format=lambda x: f'{x:.3f}') + '''
<h2>验证和证据边界</h2><ul>''' + ''.join('<li>'+html.escape(c)+'</li>' for c in verification['checks']) + '''</ul>
<p>参考 <a href="https://www.rcsb.org/structure/1PGA">1PGA</a> 有 54/55 位点与实验序列一致。不一致位点的结构分数归零但保留候选。<a href="https://www.genome.jp/ftp/db/community/aaindex/aaindex1">AAindex FAUJ880103</a> 提供归一化范德华体积。参考几何和体积仅是优先级启发式，不能证明突变后的结构或救援机制。</p>
<p>主指标仍为点值 W≥0.5；次指标为固定 WT 归一化下的 90% 条件计数区间下界≥0.5，不是生物重复的置信区间。11 个封存位点的组合标签未参与。</p>
<p>上一轮 Agnes 的 32 个背景只有 8 个计数可靠可达任务，WT 已找到全部 8 个，因此不能用该队列判断可靠救援的增益。</p>
<p>文件：<a href="protocol.json">冻结协议</a> · <a href="verification.json">验证记录</a> · <a href="comparison.csv">比较表</a> · <a href="changed_tasks.csv">变化背景</a> · <a href="query_log.csv.gz">完整查询日志</a></p></html>'''
    (OUT / 'report.html').write_text(page)
    manifest = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in [ROOT / 'scripts/structure_policy.py', ROOT / 'scripts/run_structure.py',
                          ROOT / 'scripts/verify_structure.py', ROOT / 'scripts/report_structure.py']}
    (OUT / 'code_hashes.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(table.to_string(index=False)); print('Changed tasks:'); print(changed.to_string(index=False))


if __name__ == '__main__': main()
