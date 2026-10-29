"""Write a source-grounded decision with uncertainty and test boundaries explicit."""
import hashlib,html,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/dataset_feasibility_v1'


def main():
    assert json.loads((OUT/'verification.json').read_text())['status']=='passed'
    g=pd.read_csv(OUT/'gb1_coverage_summary.csv');p18=pd.read_csv(OUT/'pdz2018_summary.csv');p26=pd.read_csv(OUT/'pdz2026_summary.csv');tasks18=pd.read_csv(OUT/'pdz2018_tasks.csv');tasks26=pd.read_csv(OUT/'pdz2026_tasks.csv')
    point18=tasks18[tasks18.regime=='all_R'];h18=tasks18[(tasks18.regime=='confident_harmful_R')&tasks18.uncertainty_available]
    raw26=tasks26[tasks26.regime=='all_R'];search26=tasks26[(tasks26.regime=='confident_harmful_R')&tasks26.searchable]
    unique=pd.read_csv(OUT/'pdz2026_unique_damage.csv')
    decision=dict(status='audit_complete_no_new_model_trials',priority_candidate='PDZ2018',formal_benchmark_migration='not_ready',
        reason_2018='dense search pools and clear point-level headroom, but a complete reliable-interval audit is unavailable for PSD95-PDZ3; observed count-supported harmful rescue backgrounds only13 in the two auditable domains, and WT finds11',
        reason_2026='only116 of1523 public-single damage assay tasks have >32 measured valid candidates; most pools19; searchable confident-harmful-R reliable rescue in only3 distinct domain/damage backgrounds',
        original_GB1='expanded all-R development pool leaves reliable reachability72 and WT@32=70 unchanged; newly disclosed full-R harmful-R audit has10 reachable versus WT@32=1',
        PDZ2018=dict(development_damage_tasks=len(point18),point_reachable=int(point18.point_reachable.sum()),WT_point32=int(point18.WT_point32.sum()),point_remaining=int((point18.point_reachable&~point18.WT_point32).sum()),candidate_median=float(point18.candidates.median()),uncertainty_unavailable_domain='pdz3',confident_harmful_R_auditable_tasks=len(h18),count_supported_reachable=int(h18.count_supported_reachable.sum()),WT_count_supported32=int(h18.WT_count_supported32.sum())),
        PDZ2026=dict(damage_assay_tasks=len(raw26),searchable_tasks=int(raw26.searchable.sum()),fraction_searchable=float(raw26.searchable.mean()),median_candidates=float(raw26.candidates.median()),confident_harmful_R_searchable_assay_tasks=len(search26),distinct_reliably_reachable_backgrounds=int(unique.reachable.sum())),
        next_action='resolve PDZ2018 PSD95-PDZ3 raw-count/energy alignment and interval calibration before formal reliable-rescue claims; a frozen point-level candidate protocol is provided but not executed; preserve untouched test proteins',
        boundaries=['2018 test shank3/zo1 never loaded','2026 ERBIN double labels not inspected','same parent protein grouped across studies','old GB1 expanded audit is post-test, not independent validation','confidence summaries from different assays/uncertainty models are not pooled as a common physical endpoint'])
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    protocol=json.loads((OUT/'audit_protocol.json').read_text())
    candidate=dict(version='pdz2018_rescue_candidate_v1',status='frozen_candidate_not_executed; reliable endpoint gate unresolved',
        source='10.7554/eLife.34300; publisher MAT identical to Dryad checksum',parent_development=['DLG4_PSD95','SNTA1_Syntrophin'],parent_test=['SHANK3','TJP1_ZO1'],
        observed_development_domains=['pdz2','pdz3','syntrophin'],unread_test_domains=['shank3','zo1'],region='nine alpha2-helix sites only; no whole-protein rescue claim',
        unit='kcal/mol WT-relative binding free-energy changes; derived experimental affinity; not fraction bound',RT_kcal_per_mol=.00198720425864083*298.15,
        public_inputs='single-mutant published energies, source identities, WT Kd and raw single counts; no double-derived coupling scores as public prior',
        oracle='published finite double-mutant deltaG; missing/noninvertible values remain unavailable; no use of stored deltadeltaG for selecting unqueried pairs',
        damage='exp(-deltaG_single/RT)<=0.2; all eligible D retained, not selected by reachable rescue',
        primary='exp(-deltaG_double/RT)>=0.5',candidate_pool='all finite measured single-edit R with finite public single energies at another scanned site; candidate >32 flag from identities/availability',
        secondary='pending validated uncertainty alignment for all development and test domains; two-domain count sensitivity is supplementary only',
        extra_regime='harmful-R subset uses only public single information and is reported separately; same subset for every comparator; threshold not tuned for rescues',
        query_budget=32,batch_size=4,initial_queries='same eight WT-ranked candidates',comparators=['WT public-single rank','fixed public-only proposal','Agnes same-information menu'],
        evaluate_test='only after final method/prompt and valid uncertainty conversion are frozen; no outcome-based data selection',
        current_decision='do not launch formal Agnes test under current reliability gaps',new_model_calls=0)
    (OUT/'pdz2018_candidate_protocol.json').write_text(json.dumps(candidate,indent=2)+'\n')
    assets=[f for f in OUT.glob('*.csv')]+[OUT/'decision.json',OUT/'audit_protocol.json',OUT/'pdz2018_candidate_protocol.json',OUT/'verification.json']
    code=[ROOT/'scripts'/f for f in ['audit_gb1_coverage.py','fetch_pdz_sources.py','prepare_pdz2026_audit.py','audit_pdz2018.py','audit_pdz2026.py','verify_dataset_audit.py','report_dataset_feasibility.py']]
    manifest={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in assets+code}
    (OUT/'decision_freeze.json').write_text(json.dumps(dict(status='frozen_audit_decision',hashes=manifest,test_outcomes_analyzed=0,new_model_calls=0),indent=2)+'\n')
    table=lambda d:d.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>救援数据集可行性审计</title><style>body{max-width:1250px;margin:35px auto;padding:0 24px;font:16px/1.75 system-ui;color:#24384c}h1,h2{color:#163c55}table{display:block;overflow-x:auto;max-width:100%;border-collapse:collapse;font-size:13px}th,td{padding:7px 9px;border-bottom:1px solid #ddd}.note{padding:15px;background:#fff1d5}a{color:#08699b}</style>
<h1>GB1覆盖 → PDZ2018 → PDZ2026：可行性审计</h1>
<p class="note"><strong>决策：2018 PDZ是优先候选，暂不启动正式Agnes评测。</strong>2018有密集候选和点值提升空间，但可靠区间评估尚不完整；2026多数背景的候选池小于预算，仅适合明确限定的文库背景子任务。没有运行新模型，测试蛋白组合结果未分析。</p>
<h2>一、GB1覆盖核对</h2><p>原始1045单突变、535917双突变；符合D≤0.2的239个全部纳入（开发188、原封存51），没有额外剔除损伤任务。原开发库341367双突变；跨拆分174695；两者均原封存位置19855，本轮未分析后者。扩池是开发D的事后诊断，新增跨位置R标签仅用于此次覆盖审计，不能重新称为独立测试。</p>'''+table(g)+'''
<p>扩池使点值可达86→88，但可靠可达仍72、WT@32仍找到70。限制单独作用不利R时出现10个计数可靠可达背景，WT@32只找到1个；这是新的探索性任务诊断，不是算法改进。</p>
<h2>二、2018 PDZ：密集，但误差映射有缺口</h2><p>在读双突变结果前按母蛋白划分。DLG4的PDZ2/PDZ3均为开发，Syntrophin为开发；SHANK3和ZO1的MAT变量未加载。仅在开发域内解释原实验换算的WT相对结合自由能，以25°C的RT换算相对结合常数，不能称为结合占比。</p>'''+table(pd.read_csv(OUT/'pdz2018_coverage.csv'))+table(p18)+'''
<p>三个开发域共207损伤背景，候选中位数123；98点值可达，WT@32找到80，尚余18。限定单独不利R后，点值可达61、WT找到48。</p>
<p>MAT提供的单/双/耦合能量符合双突变循环恒等式。原始计数的单突变标定在PDZ2/Syntrophin能复现能量；PDZ3不一致，因此不能由所给计数直接生成该域的结合常数置信区间，也不能宣称论文数据有误。潜在原因需核对原作者的重复/校准字段；本轮没有按双突变成绩调整转换。NaN换算能量也没有被臆造为极好或极差结合。</p>'''+table(pd.read_csv(OUT/'pdz2018_calibration_QC.csv'))+'''
<p>两域的计数支持只是固定WT和标定条件下的90%计数敏感性分析，未包含生物重复及标定参数不确定性。严格不利R任务中观察到13个支持救援背景，WT找11；PDZ3的可靠结果未知，不能把未知当作零。因此20背景、25%遗漏等资源门槛尚未通过完整评估。</p>
<h2>三、2026 PDZ：按损伤背景重建候选图</h2><p>通过公开Zenodo ZIP目录和HTTP范围请求，仅取得实验输入表及元数据，逐文件CRC/SHA核对。没有使用MoCHI拟合权重或推断能量。ERBIN所有配体均保留为测试；它的双突变身份可统计，fitness与sigma未进入分析。相同AA基因型只保留预选择输入计数最高的NT代表，避免把密码子等价变体当作独立生物重复。</p>'''+table(pd.DataFrame(json.loads((OUT/'pdz2026_identity_coverage.json').read_text())).drop(columns=['WT_sequence','most_connected_positions']))+'''
<p>六个开发结合实验共1523损伤“实验-背景”任务，仅116个具有>32个可用R（约7.6%）；多数候选池19，不能展示32次搜索的优势。规范分数用每块公开同义WT及单终止突变对照校准，是PCA选择读数，不是物理结合占比；区间使用提供的DiMSum sigma，固定对照归一化。</p>
<p>下表是预先按覆盖分出的可搜索子任务（没有按救援结果选任务）：</p>'''+table(p26[(p26.stratum=='searchable_gt32')&(p26.regime=='confident_harmful_R')])+'''
<p>严格单独不利R条件下，可搜索子集中仅3个不同结构域-损伤背景有区间下界达标的组合，均被WT@32找到。该结果不支持把2026数据直接作为任意损伤背景的主救援Benchmark。</p>
<h2>四、冻结的决策与下一步</h2><ol><li>保留旧GB1协议、结果和负结论，原30文件哈希一致。</li><li>优先解决2018 PDZ3的计数/能量及误差映射；点值任务协议已冻结为候选，不宣称正式可靠评测已就绪。</li><li>2026可作为背景位点受限的备选或结构研究资源；不能用总记录数代替任务覆盖。</li><li>正式模型试验需在候选可行且端点明确后另行冻结方法、提示词及测试。现有两个2018测试母蛋白与2026 ERBIN组合结果继续保留。</li></ol>
<h2>验证</h2><ul>'''+''.join('<li>'+html.escape(x)+'</li>' for x in json.loads((OUT/'verification.json').read_text())['checks'])+'''</ul>
<p><a href="decision.json">决策</a> · <a href="pdz2018_candidate_protocol.json">2018候选协议</a> · <a href="decision_freeze.json">审计冻结</a> · <a href="verification.json">交叉核验</a> · <a href="pdz2018_tasks.csv">2018任务表</a> · <a href="pdz2026_tasks.csv">2026任务表</a></p>
<p>原始来源：<a href="https://doi.org/10.7554/eLife.34300">PDZ2018论文</a>、<a href="https://doi.org/10.5061/dryad.gk4m1">Dryad</a>、<a href="https://doi.org/10.1038/s41467-026-71005-x">PDZ2026论文</a>、<a href="https://zenodo.org/records/15978782">Zenodo源档案</a>、<a href="https://github.com/lehner-lab/PDZ_homologs">作者代码</a>。</p></html>'''
    (OUT/'report.html').write_text(page)
    print(json.dumps(decision,indent=2))


if __name__=='__main__':main()
