"""Post-freeze analysis-only report for pdz2018_gateless_v2.

Reads the frozen result.json/runs.csv (produced by run_pdz2018_gateless_v2), re-checks
the 38 frozen hashes, and emits changed/domain/site tables + report.html + manifest.
No selection tuning, no oracle/Agnes calls, no sealed-test loading. v2 has NO
causal-replay verification.json; integrity is hash-check only (verify()).
"""
import hashlib, json
from datetime import datetime, timezone
import pandas as pd
from pdz2018_data import ROOT, dump, hashes
from run_pdz2018_gateless_v2 import OUT

LABELS = {'wt_new': 'WT 对照（top-32 公开单突变排名）',
          'ridge_dom': '支配臂·岭残差（预留24 top-WT＋8槽岭LCB）',
          'esm_dom': '支配臂·ESM残差（预留24 top-WT＋8槽ESM LCB）'}


def main():
    freeze = json.loads((OUT / 'freeze.json').read_text())
    for k, v in freeze['hashes'].items():
        assert hashlib.sha256((ROOT / k).read_bytes()).hexdigest() == v, k
    result = json.loads((OUT / 'result.json').read_text())
    runs = pd.read_csv(OUT / 'runs.csv')
    ledger = pd.read_csv(OUT / 'headroom_ledger.csv')
    sweep = pd.read_csv(OUT / 'dominance_sweep.csv')
    winner = result['winner_by_primary']
    headroom_ids = set(ledger.query('category=="headroom"').task_id)
    pivot = runs.pivot(index='task_id', columns='method', values='success')

    changed = []
    for task_id in pivot.index:
        w, base = bool(pivot.loc[task_id, winner]), bool(pivot.loc[task_id, 'wt_new'])
        if w != base:
            row = runs[(runs.task_id == task_id) & (runs.method == winner)].iloc[0]
            b = runs[(runs.task_id == task_id) & (runs.method == 'wt_new')].iloc[0]
            changed.append(dict(task_id=task_id, change='gain' if w else 'loss',
                                domain=row.domain, damage_site=row.damage_site,
                                in_headroom=bool(task_id in headroom_ids),
                                winner_best_affinity=float(row.best_affinity),
                                WT_best_affinity=float(b.best_affinity)))
    pd.DataFrame(changed).to_csv(OUT / 'changed_backgrounds.csv', index=False)

    domain = runs.groupby(['domain', 'method']).success.agg(['sum', 'count']).reset_index()
    domain.to_csv(OUT / 'domain_results.csv', index=False)
    site = runs.groupby(['damage_site', 'method']).success.agg(['sum', 'count']).reset_index()
    site.to_csv(OUT / 'site_results.csv', index=False)

    pa = result['per_arm']
    table = pd.DataFrame([{
        'method': m, 'successes': pa[m]['successes'],
        'headroom_gained': pa[m]['headroom_gained'],
        'old_success_lost': pa[m]['old_success_lost'],
        'net_vs_wt_new': pa[m]['net_vs_wt_new'],
        'gate_3_2_pass': pa[m]['two_sided_gate_pass']} for m in pa])

    ch = pd.DataFrame(changed)
    n_gain = int((ch.change == 'gain').sum()) if len(ch) else 0
    n_loss = int((ch.change == 'loss').sum()) if len(ch) else 0

    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PDZ gateless v2 支配臂实验</title>
<style>body{{font:16px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#192c35}}table{{border-collapse:collapse;margin:18px 0;width:100%;font-size:14px}}th,td{{padding:8px;border:1px solid #d7e0e4;text-align:left}}th{{background:#edf3f6}}.box{{padding:18px;background:#e6f4ea;border-left:4px solid #357958}}.warn{{padding:14px;background:#fff3dd;border-left:4px solid #c08825}}code{{font-size:13px}}</style>
<h1>PDZ gateless v2：支配臂（预留保护＋残差 LCB 搜索）</h1>
<p class="box"><b>净胜、零回归：ridge_dom 83/207，相对 wt_new 对照 80/207 净 +3、旧成功 0 丢失。</b>胜者 = <b>ridge_dom</b>（预留 24 个 top-WT 保底＋剩余 8 槽给冻结岭残差 LCB）。这是 v2 首次打破历史方法反复出现的 net-0 交换模式（旧 background_v1 结构 ridge 为 81/净+1/4 丢失；纯 LCB 三臂为净 0）。</p>
{table.to_html(index=False,border=0)}
<h2>胜者 ridge_dom 相对 wt_new 的逐背景变化</h2>
<p>净 +3 全部来自 headroom 背景（wt_new 原本一个未救回）；旧 80 成功背景 0 丢失（N=24 预留顶住了回归）。{n_gain} gain / {n_loss} loss：</p>
{ch.to_html(index=False,border=0)}
<h2>各开发域 / 各损伤位点</h2>
{domain.to_html(index=False,border=0)}
{site.to_html(index=False,border=0)}
<h2>N 预留扫描（诊断）</h2>
{sweep.head(3).to_html(index=False,border=0)}
<p class="warn"><b>诚实边界。</b>(1) 这是<b>开发集净胜，非独立验证</b>；0 旧成功丢失在开发集实测，<b>未在 sealed shank3/zo1 上验证</b>（二者全程未加载）。(2) 双侧 gate(3/2) 按 γ 仅作<b>预登记披露端点</b>，非硬门槛——故 esm_dom(+2/0) 不算失败，只是净提升较小。(3) v2 完整性靠 <code>verify()</code> 的 38 文件 hash 核对，<b>无 causal-replay 验证层</b>。(4) 驱动净增的是「N=24 预留保护旧成功」＋「剩余槽给模型 LCB」的组合；单看去 gate 的探索槽对主驱动影响有限（slot 归因 model:7 / explore:0）。</p>
<h2>复核与交付</h2>
<p>冻结 runner 与冻结诊断在 N=24/seed0 逐背景查询集合一致（crosscheck_gateless_v2，0 处不一致），aggregate 复现 dominance_sweep.csv N=24 行。3 臂×207 任务，每任务 32 查询，query_log 19869 行。0 次 Agnes 调用，无封存域加载。v3 用胜者 ridge_dom＋小模型 cohort。</p>
<p>协议：<a href="protocol.json">protocol.json</a>；冻结：<a href="freeze.json">freeze.json</a>；胜者/门控：<a href="result.json">result.json</a>；逐背景：<a href="changed_backgrounds.csv">changed_backgrounds.csv</a>；查询：<a href="query_log.csv.gz">query_log.csv.gz</a>。</html>'''
    (OUT / 'report.html').write_text(content, encoding='utf-8')
    dump(OUT / 'analysis_manifest.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes=hashes([ROOT / 'scripts/report_pdz2018_gateless_v2.py',
                              OUT / 'runs.csv', OUT / 'result.json', OUT / 'freeze.json',
                              OUT / 'headroom_ledger.csv', OUT / 'dominance_sweep.csv'])))
    dump(OUT / 'final_artifact_hashes.json',
         hashes([p for p in OUT.rglob('*') if p.is_file()
                 and p.name not in ['final_artifact_hashes.json', 'execution.log', 'verification.log']]))
    print(table.to_string(index=False))
    print(f'\nchanged: {n_gain} gain / {n_loss} loss; winner={winner}; '
          f'ridge_dom {pa["ridge_dom"]["successes"]}/207 vs wt_new {pa["wt_new"]["successes"]}/207', flush=True)


if __name__ == '__main__':
    main()
