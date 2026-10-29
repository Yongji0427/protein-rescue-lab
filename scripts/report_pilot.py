"""Produce exploratory diagnostic tables, a figure, and a self-contained report."""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_pilot import count_interval


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/gb1_pilot_v1"


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    audit = json.loads((OUT / "audit.json").read_text())
    summary = pd.read_csv(OUT / "summary.csv")
    tasks = pd.read_csv(OUT / "development_tasks.csv")
    runs = pd.read_csv(OUT / "runs.csv")
    examples = pd.read_csv(OUT / "wt_examples.csv")
    singles = pd.read_csv(ROOT / "data/public_reference/singles.csv").set_index("mutant")
    raw = pd.read_csv(ROOT / "data/evaluator_only/gb1_raw.csv")
    held = protocol["holdout_full_sequence_positions"]
    raw = raw.loc[raw["Mut2 Position"].notna() & ~(raw["Mut1 Position"] + 226).isin(held) & ~(raw["Mut2 Position"] + 226).isin(held)].copy()
    parts = raw.mutant.str.split(":", expand=True)
    raw["r1"], raw["r2"] = parts[0], parts[1]
    raw["ci_low"], raw["ci_high"] = count_interval(raw)
    groups1 = {key: group for key, group in raw.groupby("r1", sort=False)}
    groups2 = {key: group for key, group in raw.groupby("r2", sort=False)}
    empty = raw.iloc[0:0]
    diagnostics, misses, background_examples = [], [], []
    for task in tasks.itertuples():
        table = pd.concat([groups1.get(task.damage, empty), groups2.get(task.damage, empty)]).copy()
        table["rescue"] = np.where(table.r1 == task.damage, table.r2, table.r1)
        table["single_W"] = table.rescue.map(singles.W)
        table["single_lnW"] = table.rescue.map(singles.lnW)
        table = table.sort_values(["single_lnW", "rescue"], ascending=[False, True])
        table["wt_rank"] = np.arange(1, len(table) + 1)
        rescued = table.loc[table.W >= .5]
        assert len(rescued) == task.rescue_candidates
        background = rescued.loc[(rescued.single_W < 1) & (rescued.ci_low >= .5)]
        if len(background):
            best_background = background.sort_values("ci_low", ascending=False).iloc[0]
            background_examples.append({"damage": task.damage, "rescue": best_background.rescue,
                "damage_W": task.damage_W_raw, "rescue_single_W": best_background.single_W,
                "rescue_single_ci_high": singles.loc[best_background.rescue, "count_ci_high"],
                "combination_W": best_background.W, "combination_ci_low": best_background.ci_low,
                "WT_rank": int(best_background.wt_rank), "discovery_source": "evaluator_post_run_audit"})
        diagnostics.append({"damage": task.damage,
            "sign_reversal_rescues": int((rescued.single_W < 1).sum()),
            "count_robust_sign_reversal_rescues": int(((rescued.single_W < 1) & (rescued.ci_low >= .5)).sum()),
            "rescues_with_input_below_10": int((rescued["Input Count"] < 10).sum()),
            "best_rescue_WT_rank": int(rescued.wt_rank.min()) if len(rescued) else None})
        example = examples.loc[examples.damage == task.damage].iloc[0]
        if task.rescue_candidates and not example.success:
            best = rescued.sort_values("W", ascending=False).iloc[0]
            misses.append({"damage": task.damage, "rescue": best.rescue, "damage_W_raw": task.damage_W_raw,
                "rescue_single_W": best.single_W, "combination_W": best.W,
                "count_ci_low": best.ci_low, "count_ci_high": best.ci_high,
                "WT_rank": int(best.wt_rank), "Input_Count": int(best["Input Count"]),
                "Selection_Count": int(best["Selection Count"]), "count_robust_rescue": bool(best.ci_low >= .5)})
    diagnostics = pd.DataFrame(diagnostics)
    misses = pd.DataFrame(misses)
    diagnostics.to_csv(OUT / "task_diagnostics.csv", index=False)
    misses.to_csv(OUT / "wt_missed_rescues.csv", index=False)
    pd.DataFrame(background_examples).to_csv(OUT / "background_dependent_examples.csv", index=False)
    counts = {
        "tasks_with_sign_reversal_rescue": int((diagnostics.sign_reversal_rescues > 0).sum()),
        "tasks_with_count_robust_sign_reversal_rescue": int((diagnostics.count_robust_sign_reversal_rescues > 0).sum()),
        "WT_missed_tasks": len(misses), "WT_missed_count_robust_tasks": int(misses.count_robust_rescue.sum()),
    }
    (OUT / "diagnostic_summary.json").write_text(json.dumps(counts, indent=2) + "\n")
    curves = pd.read_csv(OUT / "curves.csv")
    labels = {"random": "Random (100 repeats/task)", "wt_single_rank": "WT single-mutant ranking"}
    colors = {"random": "#8a8f98", "wt_single_rank": "#346b9d"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    for method in labels:
        frame = curves.loc[(curves.method == method) & (curves["query"] % 4 == 0)]
        axes[0].plot(frame["query"], frame.success_rate * 100, "o-", color=colors[method], label=labels[method])
        axes[1].plot(frame["query"], frame.mean_best_W, "o-", color=colors[method])
    exact = curves.loc[(curves.method == "random") & (curves["query"] % 4 == 0)]
    axes[0].plot(exact["query"], exact.random_exact_success * 100, ":", color="#24272b", label="Random: exact expectation")
    ceiling = audit["reachable_development_tasks"] / audit["development_tasks"] * 100
    axes[0].axhline(ceiling, color="#bd7e32", ls="--", label=f"Pool reachability ceiling ({ceiling:.1f}%)")
    axes[0].set(xlabel="New double-mutant label queries", ylabel="Rescue success (%)", ylim=(0, 53), title="A  Success at 50% WT")
    axes[0].legend(fontsize=8, loc="center right")
    axes[1].set(xlabel="New double-mutant label queries", ylabel="Mean best WT-relative binding", title="B  Best binding found")
    axes[1].axhline(.5, color="#bd7e32", ls="--", lw=1)
    axes[1].text(32, .51, "Rescue threshold", ha="right", color="#946528", fontsize=8)
    bins = [int((tasks.rescue_candidates == 0).sum()), int(tasks.rescue_candidates.between(1, 5).sum()), int(tasks.rescue_candidates.between(6, 20).sum()), int((tasks.rescue_candidates > 20).sum())]
    bars = axes[2].bar(["0", "1–5", "6–20", ">20"], bins, color=["#8a8f98", "#b9cde0", "#719bbb", "#346b9d"])
    axes[2].bar_label(bars, padding=3)
    axes[2].set(xlabel="Rescue candidates in permitted pool", ylabel="Development tasks", title="C  Task reachability")
    fig.suptitle("GB1 preliminary replay | 188 development tasks | held-out double labels unevaluated", fontsize=13)
    fig.savefig(OUT / "pilot_results.png", dpi=180)
    fig.savefig(OUT / "pilot_results.svg")
    plt.close(fig)

    headline = summary[["method", "budget", "success_rate", "count_robust_success_rate", "mean_best_W", "median_best_W"]].copy()
    headline["method"] = headline.method.map(labels)
    for col in ["success_rate", "count_robust_success_rate"]:
        headline[col] = headline[col].map(lambda x: f"{100*x:.2f}%")
    for col in ["mean_best_W", "median_best_W"]:
        headline[col] = headline[col].map(lambda x: f"{x:.3f}")
    headline.columns = ["方法", "新增查询数", "点估计救援率", "计数区间确认救援率", "最佳结合均值 / WT", "最佳结合中位数 / WT"]
    by_difficulty = tasks[["damage", "rescue_candidates"]].copy()
    by_difficulty["stratum"] = pd.cut(by_difficulty.rescue_candidates, [-1, 0, 5, 20, np.inf], labels=["0", "1–5", "6–20", ">20"])
    strata = runs.loc[runs.budget == 32].merge(by_difficulty, on="damage")
    strata_table = strata.groupby(["stratum", "method"], observed=True).agg(success_rate=("success", "mean"), task_count=("damage", "nunique")).reset_index()
    strata_table.to_csv(OUT / "difficulty_strata.csv", index=False)
    case = examples.loc[(examples.combination_count_ci_low >= .5) & (~examples.damage_at_floor) & (examples.damage_count_ci_high <= .2)].sort_values("combination_count_ci_low", ascending=False).iloc[0]
    controls = pd.DataFrame({"变体": ["WT（归一化参考）", case.damage, case.rescue, case.combination], "结合表现 / WT": [1.0, case.damage_W_raw, case.rescue_single_W, case.combination_W]})
    controls.to_csv(OUT / "four_controls_example.csv", index=False)
    sources = json.loads((ROOT / "data/provenance/sources.json").read_text())
    sections = [
        "<h1>GB1 蛋白救援初步实验</h1><p class='tag'>探索性开发实验 · 公开历史实验数据回放 · 未接入 Agnes · 预留评测组合标签未评测</p>",
        f"<p><b>WT 单突变排序是很强的基线。</b>32 次查询下，成功率为 42.55%（80/188），随机搜索为 12.80%。候选池内有可行救援的开发任务为 86/188，因此 WT 排序已找到其中 80 个。上述结论只适用于当前开发任务与候选池。</p>",
        "<img src='pilot_results.png' alt='GB1 preliminary baseline results'>",
        "<h2>固定协议与数据审计</h2><ul><li>损伤：结合表现 ≤ 0.2 WT；救援：组合 ≥ 0.5 WT；更强恢复：≥ 1.0 WT。</li><li>一个固定损伤、另一个位置的一次替换；所有单突变参考免费可见，只对新组合标签计费。</li><li>预算 8/16/32，每批 4 个；首次成功按整批成本计数；成功后仍运行至 32 次。</li><li>随机搜索每任务 100 个固定种子；WT 排序按单突变 lnW 降序，平分按变体 ID。</li></ul>",
        f"<p>官方数据包含 {audit['raw_records']:,} 条记录、1,045 条单突变。239 个损伤背景中，188 个属于开发位置，51 个位于预留位置。开发候选池仅包含双方均位于开发位置的组合，每任务 812–817 个候选；0 个任务因覆盖不足被排除。102 个不可达任务仍保留在总体分母中。</p>",
        f"<p>按种子 20261008 在 55 个被测位置中预留 11 个位置：{html.escape(str(protocol['holdout_full_sequence_positions']))}。开发过程中没有评测涉及这些位置的组合标签。编号经原表字段核验：全长位置 = 原论文结构域位置 + 226；被测区域索引 = 全长位置 − 227。结构残基映射仍需实际结构比对。</p>",
        "<h2>总体结果</h2>", headline.to_html(index=False, escape=True),
        "<p>分母均为 188 个开发任务。随机数值为 100 次搜索重复的平均，不是实验重复。图中的随机理论期望使用每个任务实际候选数和可救援候选数的超几何分布精确计算，仅由评测端使用。</p>",
        "<h2>计数不确定性与检测下限</h2>",
        f"<p>lnW = log(max(W, 0.01))，故 exp(lnW)=0.01 表示下限删失，不能解释为精确的 1% 结合。188 个开发损伤中，{audit['floor_damage_tasks']} 个位于下限；这些任务只有 3 个存在点估计救援，0 个存在计数区间确认的救援。相对结合表现来自实验富集指标，不能直接称为细胞功能、折叠比例或治疗效果。</p>",
        "<p>原始表保留 Input Count 和 Selection Count，W 等于其计数比。附加敏感性分析在独立 Poisson 计数和固定 WT 归一化假设下，计算 90% 精确条件计数比区间，组合下界 ≥ 0.5 才记为计数区间确认救援。它不包含 WT 归一化误差或生物学重复方差，也不等同于复现原论文的高置信度筛选。</p>",
        f"<p>候选池内计数区间确认可达任务为 {audit['robust_reachable_tasks']}/188；WT 排序找到 70 个。若同时要求损伤计数区间上界 ≤ 0.2，187 个任务合格，WT 排序确认救援率为 69/187=36.90%。</p>",
        "<h2>WT 排序漏掉的六个开发背景</h2>", misses.to_html(index=False, float_format=lambda x: f"{x:.5g}"),
        "<p>这些是完成主实验后的开发集诊断，不是按结果筛选出的正式评测集。六个遗漏背景中，仅一个的最佳救援通过计数区间检查；其他候选位于阈值附近或计数不足。后续策略应先围绕可靠、稀有的补偿开展开发，再用预留位置评测泛化。</p>",
        f"<p>全池事后审计中，{counts['tasks_with_sign_reversal_rescue']} 个开发背景存在补偿单独低于 WT、组合达到救援标准的候选，其中 {counts['tasks_with_count_robust_sign_reversal_rescue']} 个背景的组合救援通过计数区间检查。这是评测端审计发现，不等于搜索方法查询发现，也不构成物理机制或新发现的证据。</p>",
        "<h2>四组对照示例</h2>", controls.to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        f"<p>该组合的计数区间为 [{case.combination_count_ci_low:.3f}, {case.combination_count_ci_high:.3f}]。它是基线发现的开发示例，补偿单独出现时已明显增强结合，不能将此案例描述为只在损伤背景有效的补偿。</p>",
        "<h2>验证、复现与证据边界</h2>",
        "<p>已核验 607,616 条查询日志、18,988 次搜索运行：无重复新增计费、无同位点编辑、固定损伤保留、预算正确、无预留位置组合进入评测。342,412 个开发或公开单突变分数与官方处理版本一致，最大差异小于 2×10⁻¹⁵；抽查 1,000 条完整序列与变体描述一致。原始及处理数据均保存 SHA256。</p>",
        "<p>策略接口只接收候选身份和单突变参考，标签由 evaluator 对象返回。当前为受控基线脚本，没有为自由使用 shell 的智能体实现操作系统级文件隔离；接入 Agnes 前必须进一步限制标签文件权限和可用工具。</p>",
        "<pre>python3 scripts/download_data.py\npython3 scripts/run_pilot.py\npython3 scripts/verify_pilot.py\npython3 scripts/report_pilot.py</pre>",
        f"<p>数据版本 ProteinGym v1.3；参考元数据提交 {html.escape(sources['metadata_commit'])}；协议 SHA256 {audit['protocol_sha256']}。主实验耗时约 {audit['wall_seconds']:.1f} 秒，不含数据下载和后续独立验证。</p>",
        "<p>数据来源：<a href='https://github.com/OATML-Markslab/ProteinGym'>ProteinGym 官方仓库</a>；<a href='https://doi.org/10.1016/j.cub.2014.09.072'>Olson et al., 2014</a>。本轮并未验证 Agnes 搜索优势、结构机制或新变体的湿实验效果。</p>",
    ]
    report = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>GB1 初步实验报告</title><style>body{font-family:system-ui,sans-serif;max-width:1180px;margin:40px auto;padding:0 24px;color:#243343;line-height:1.7}h1{font-size:30px}h2{margin-top:32px;font-size:21px}.tag{color:#687789}img{width:100%;height:auto}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}th,td{padding:9px 12px;border-bottom:1px solid #dce4eb;text-align:left}th{background:#edf2f7}pre{background:#edf2f7;padding:18px;overflow:auto}a{color:#346b9d}</style><body>" + "\n".join(sections) + "</body></html>"
    (OUT / "report.html").write_text(report)
    print(json.dumps(counts, indent=2))
    print(misses.to_string(index=False))
    print(controls.to_string(index=False))
    print("Report:", OUT / "report.html")


if __name__ == "__main__":
    main()
