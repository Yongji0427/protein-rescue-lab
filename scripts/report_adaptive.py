"""Report all frozen adaptive results, including lack of primary improvement."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/gb1_adaptive_v1"
PILOT = ROOT / "experiments/gb1_pilot_v1"


def main():
    adaptive = pd.read_csv(OUT / "summary.csv")
    baseline = pd.read_csv(PILOT / "summary.csv")
    summary = pd.concat([baseline, adaptive], ignore_index=True)
    headline = summary.loc[summary.budget.isin([8, 16, 32]), ["method", "budget", "success_rate", "count_robust_success_rate", "mean_best_W"]]
    headline.to_csv(OUT / "comparison.csv", index=False)
    tasks = pd.read_csv(PILOT / "development_tasks.csv")
    per_task = pd.read_csv(OUT / "per_task.csv")
    base_runs = pd.read_csv(PILOT / "runs.csv")
    paired = per_task.merge(tasks[["damage", "full_position"]], on="damage")
    task_base = base_runs.query("method=='wt_single_rank'").rename(columns={"success": "wt_success", "count_robust_success": "wt_robust"})
    paired = paired.merge(task_base[["damage", "budget", "wt_success", "wt_robust"]], on=["damage", "budget"])
    paired["success_delta_vs_WT"] = paired.success_rate - paired.wt_success
    paired["robust_delta_vs_WT"] = paired.count_robust_success_rate - paired.wt_robust
    paired.to_csv(OUT / "paired_task_differences.csv", index=False)
    differences = []
    rng = np.random.default_rng(20261009)
    for budget in [8, 16, 32]:
        pivot = per_task.loc[per_task.budget == budget].pivot(index="damage", columns="method", values="success_rate")
        contrast_direct = pivot.contrast - pivot.direct
        blocks = pd.DataFrame({"damage": contrast_direct.index, "delta": contrast_direct.values}).merge(tasks[["damage", "full_position"]], on="damage").groupby("full_position").delta.agg(["sum", "count"])
        draws = rng.integers(0, len(blocks), size=(5000, len(blocks)))
        sampled = blocks["sum"].to_numpy()[draws].sum(1) / blocks["count"].to_numpy()[draws].sum(1)
        differences.append({"budget": budget, "contrast_minus_direct_pp": 100 * contrast_direct.mean(),
            "position_bootstrap_95_low_pp": 100 * np.quantile(sampled, .025),
            "position_bootstrap_95_high_pp": 100 * np.quantile(sampled, .975)})
    pd.DataFrame(differences).to_csv(OUT / "contrast_ablation_differences.csv", index=False)
    curves = pd.read_csv(PILOT / "curves.csv")
    robust_curves = base_runs.groupby(["method", "budget"]).count_robust_success.mean().reset_index()
    labels = {"random": "Random", "wt_single_rank": "WT ranking", "direct": "Direct feedback", "contrast": "Contrast driven"}
    colors = {"random": "#93979c", "wt_single_rank": "#376d9b", "direct": "#8b74a2", "contrast": "#bc783a"}
    styles = {"random": ":", "wt_single_rank": "--", "direct": "-.", "contrast": "-"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for method in labels:
        if method in ["contrast", "direct"]:
            data = adaptive.loc[adaptive.method == method]
            axes[0].plot(data.budget, data.success_rate * 100, styles[method], color=colors[method], lw=1.9, label=labels[method])
            axes[1].plot(data.budget, data.count_robust_success_rate * 100, styles[method], color=colors[method], lw=1.9)
        else:
            data = curves.loc[(curves.method == method) & (curves["query"] % 4 == 0)]
            axes[0].plot(data["query"], data.success_rate * 100, styles[method], color=colors[method], lw=1.5, label=labels[method])
            data = robust_curves.loc[robust_curves.method == method]
            axes[1].plot(data.budget, data.count_robust_success * 100, styles[method], color=colors[method], lw=1.5)
    axes[0].axhline(86 / 188 * 100, color="#b5b5b5", lw=1)
    axes[1].axhline(72 / 188 * 100, color="#b5b5b5", lw=1)
    for axis in axes:
        axis.set(xlabel="New combination-label queries", ylabel="Success (%)", xlim=(3, 33), ylim=(0, 50))
        axis.set_xticks([4, 8, 16, 24, 32])
    axes[0].set_title("A  Rescue at 50% WT: no primary improvement")
    axes[1].set_title("B  Count-interval-confirmed rescue")
    axes[0].legend(loc="center right", frameon=False)
    fig.suptitle("GB1 adaptive development replay | 188 tasks | 100 repeats | no Agnes calls", fontsize=12)
    fig.savefig(OUT / "adaptive_results.png", dpi=180)
    fig.savefig(OUT / "adaptive_results.svg")
    plt.close(fig)

    names = {"random": "随机搜索", "wt_single_rank": "WT 单突变排序", "direct": "直接组合反馈（消融）", "contrast": "对照驱动自适应搜索"}
    display = headline.copy()
    display.method = display.method.map(names)
    for col in ["success_rate", "count_robust_success_rate"]:
        display[col] = display[col].map(lambda x: f"{100*x:.3f}%")
    display.mean_best_W = display.mean_best_W.map(lambda x: f"{x:.4f}")
    display.columns = ["方法", "查询预算", "救援成功率", "计数区间确认救援率", "最佳结合均值 / WT"]
    delta = adaptive.query("method=='contrast' and budget==32").success_rate.iloc[0] - baseline.query("method=='wt_single_rank' and budget==32").success_rate.iloc[0]
    verification = json.loads((OUT / "verification.json").read_text())
    audit = json.loads((OUT / "audit.json").read_text())
    agnes = json.loads((OUT / "agnes_status.json").read_text())
    sections = [
        "<h1>对照驱动自适应搜索：第一版实验结果</h1><p class='tag'>开发探索 · 已观察基线后提出的策略 · 预留评测位置继续封存 · Agnes 尚未运行</p>",
        f"<p><b>主指标没有提升。</b>32 次查询下，对照驱动版本成功率 42.053%，直接组合反馈消融版 42.048%，WT 排序基线 42.553%。对照驱动比 WT 基线低 {abs(delta)*100:.3f} 个百分点，比消融仅高 0.005 个百分点。这不支持当前实现能提高总体救援成功率。</p>",
        "<img src='adaptive_results.png' alt='Adaptive search comparison'>",
        "<h2>方法的具体实现</h2><p>每个任务的前 8 次查询沿用 WT 单突变排序。之后每批 4 个候选：1 个最高剩余 WT 排序候选，2 个预测组合表现最高的候选，1 个在尚未探索位置的随机候选。四个选择先全部提交，再同时处理反馈；每次运行独立，不跨损伤背景共享组合标签。</p>",
        "<p>候选相似性固定为：同位置 1.0、替换氨基酸同组 0.25、全局权重 0.05；氨基酸组为 AVLIM / FWY / STNQ / KRH / DE / CGP。这是粗略搜索特征，不是物理机制证据。收缩参数固定为 1.0，没有进行参数扫描或按失败任务加入特例。</p>",
        "<p>以 WT lnW=0 为参考，先验 pᵢ=y(D)+y(Rᵢ)。对照版本用已查询残差 Eⱼ=y(D+Rⱼ)−pⱼ 更新：<code>ŷᵢ=pᵢ+ΣKᵢⱼEⱼ/(1+ΣKᵢⱼ)</code>。消融版本直接平滑组合测量：<code>ŷᵢ=(pᵢ+ΣKᵢⱼy(D+Rⱼ))/(1+ΣKᵢⱼ)</code>。</p>",
        "<p>两版本使用相同候选池、初始八次查询、相似性、收缩参数、探索配额和随机种子。消融仍保留共同的 WT 单突变先验，检验的是显式背景对照残差变换，不是完全移除单突变信息。</p>",
        "<h2>全部主比较</h2>", display.to_html(index=False),
        "<p>分母为全部 188 个开发任务，包括 102 个当前候选池内不可达任务。随机和自适应方法每任务运行 100 次；这些重复是搜索随机性，不是独立湿实验重复。预留的 11 个位置、51 个损伤背景涉及的双突变标签未评测。</p>",
        "<h2>消融与次要指标</h2>", pd.DataFrame(differences).to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        "<p>位置分组 bootstrap 仅描述本开发集合上的配对差异，不能替代预留位置评测或证明对新蛋白的泛化。32 次查询下，对照版本与消融版的差异极小，不能主张对照变换带来稳定成功率增益。</p>",
        "<p>次要的计数区间确认救援指标在 16 次查询时，对照版本为 37.170%，消融为 36.197%，WT 排序为 36.170%；但在 32 次时三者几乎一致。因此只能记录部分确认结果出现更早的开发现象，不能以此替换未改善的主指标。计数区间依赖固定归一化与 Poisson 假设，不涵盖生物学重复误差。</p>",
        "<h2>判断与下一步</h2><p>第一版粗粒度相似性与残差平滑没有解决 WT 排序之外的稀有补偿。保留本版本和强基线，不据此宣称智能体或对照方法有效。下一步应先检查未查询候选的预测排序能否识别稀有补偿，以及探索配额是否足够；Agnes 决策版本必须单独运行、单独报告。正式评测位置保持封存，不能在当前开发失败后用于继续调参。</p>",
        f"<h2>验证与复现</h2><p>验证状态 {verification['status']}。已检查 {verification['queries']:,} 条查询的预算、唯一性、固定损伤和预留位置排除；另对两个方法在所有任务的种子 0 运行逐批重放，共重建 {verification['replay_batches']:,} 个批次，只使用之前已揭示的反馈。公式计算与日志汇总也通过独立核验。主实验约 {audit['wall_seconds']:.1f} 秒。</p>",
        "<pre>OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/run_adaptive.py\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/verify_adaptive.py\npython3 scripts/report_adaptive.py</pre>",
        f"<p>协议 SHA256：{audit['protocol_sha256']}；策略 SHA256：{audit['policy_sha256']}。</p>",
        f"<h2>Agnes 运行状态</h2><p>模型调用次数：{agnes['model_calls']}。已知 AGH 账号配置文件当前不存在，本地工作台返回 HTTP {agnes['local_AGH_web_http_status']}。需要用户提供已配置账号的文件或应用位置，未读取或打印密钥。上述固定规则结果不代表 Agnes 决策版本的表现。</p>",
        "<p>数据、计数区间和编号说明沿用 <a href='../gb1_pilot_v1/report.html'>第一轮基线报告</a>。原始 idea 和基线结果均保留。</p>",
    ]
    document = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>自适应搜索第一版报告</title><style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#243343;line-height:1.7}h1{font-size:29px}h2{font-size:21px;margin-top:30px}.tag{color:#687789}img{width:100%}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}td,th{padding:9px 12px;border-bottom:1px solid #dce4eb;text-align:left}th,pre{background:#edf2f7}pre{padding:18px;overflow:auto}code{background:#edf2f7}a{color:#346b9d}</style><body>" + "\n".join(sections) + "</body></html>"
    (OUT / "report.html").write_text(document)
    print(headline.to_string(index=False))
    print(pd.DataFrame(differences).to_string(index=False))
    print("Report:", OUT / "report.html")


if __name__ == "__main__":
    main()
