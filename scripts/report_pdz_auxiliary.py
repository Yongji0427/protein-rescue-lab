"""Concrete admission decisions for auxiliary functional information."""
import json
import pandas as pd
from pdz2018_data import dump,hashes
from audit_pdz_auxiliary import OUT,check


def main():
    check();assert not (OUT/'result.json').exists()
    verification=json.loads((OUT/'verification.json').read_text());assert verification['status']=='passed'
    files=pd.read_csv(OUT/'file_audit.csv');paired=pd.read_csv(OUT/'paired_phenotypes.csv');multi=pd.read_csv(OUT/'multi_ligand_overlap.csv');overlap=pd.read_csv(OUT/'overlap_with2018.csv');splits=pd.read_csv(OUT/'split_eligibility.csv');maps=json.loads((OUT/'sequence_mapping.json').read_text())
    replicates=[c for c in files if c.startswith('replicate_r')]
    decisions=[dict(source='PSD95-PDZ2 abundance + CRIPT/NMDAR2A binding',status='eligible_for_fold_safe_auxiliary_pilot',
        reason='Two known ligand sequences, paired abundance/binding, three replicate fields; sameparent/position/genotype exclusion required'),
        dict(source='PSD95-PDZ3 abundance + CRIPT binding',status='eligible_with_construct_annotation',
        reason='Actualtable WT matchesauthorFASTA; metadata hasterminalE instead of stop. Sharedscan9 mapping valid. Faure2022 data reused in2026, not independent validation'),
        dict(source='NHERF3-PDZ1 abundance + UHMK1/DAP1 binding',status='eligible_for_cross_domain_auxiliary_supervision',
        reason='Two ligand sequences available; sharedhomologous9site map verified; sameparentPDZK1 groupswithPDZ2'),
        dict(source='NHERF3-PDZ2 abundance + CEP164 binding',status='paired_phenotypes_available_ligand_sequence_conditional',
        reason='Ligand metadata XNRVKVYRF containsX; do not invent residue. Abundance andassay-ID-based pairing available; sequence-conditioned admission waitsfor clarification'),
        dict(source='ERBIN / SHANK3 / ZO1 outcomes',status='excluded_sealed',reason='No outcome files read'),
        dict(source='Author fitted MoCHI energies/weights',status='excluded_as_training_or_validation_labels',reason='Globally fitted artifacts may contain evaluation information; refit inside permittedfolds if needed'),
        dict(source='2026 collection as replacement32-query benchmark',status='not_ready',reason='Priorcoverage audit:116/1523 tasks have>32 candidates; mostmedian19. Keep asauxiliary information, not unqualified benchmark migration')]
    plan=dict(next_minimum_pilot='Paired PSD95-PDZ2 abundance/CRIPТ/NMDAR2A auxiliary-information test; no training started',
        input_fields=['sourcehash','study_family','parent_gene','domain','actual_construct_sequence','ligand_sequence','phenotype','block','nt_sequence','canonical_AA_mutations','replicate_counts','replicate_fitness','replicate_sigma','aggregate_fitness','aggregate_sigma'],
        supervised_regime='Strict heldalignedsite masksapply toallauxiliary single/double phenotypes andligands; leave-domain andleave-parent groupsdefinedseparately',
        enhanced_reference_regime='If external single abundance/binding observations are supplied for held mutations, declare anew reference-access regime and give every baseline identical access; report information gain separately fromalgorithm gain',
        model_targets='Use assay-specific phenotype outputs and uncertainty; abundance is aproxy constrainingfolding, not adirectfoldingenergy label; bindingPCA is not literal affinity. Do not simply subtracttwo assays toinferbindingenergy.',
        controls=['originalinformation baseline','same auxiliaryinformation simplecontrol','same auxiliaryinformation mechanistic/sharedrepresentation candidate'],
        endpoints='Keep original2018 pointandconditionalcount metrics separate; newdataset cannotrepair2018 PDZ3 replicate/calibration mapping',
        validation='Parent/domain/genotype groups crossstudyandligand; publicalignmentonly; train-only normalizationandmechanisticparameter fitting; no author globalweights',
        unresolved=['2018 fullassay constructs andreplicate/calibration gaps persist','CEP164 X residue/padding meaning unresolved','Only2 development parentproteins limit crossprotein claims'])
    dump(OUT/'admission_decisions.json',decisions);dump(OUT/'integration_plan.json',plan)
    result=dict(status='audit_complete_no_training',development_files=18,domains=4,parent_proteins=2,binding_conditions=6,abundance_conditions=4,
        paired_phenotypes=paired.to_dict('records'),multi_ligand=multi.to_dict('records'),shared2018_patterns=overlap.to_dict('records'),
        descriptive_replicate_correlation_range=[float(files[replicates].min().min()),float(files[replicates].max().max())],
        metadata_sequence_mismatches=[r['domain'] for r in maps if not r['metadata_matches']],
        ligand_sequence_unknown=paired.loc[~paired.ligand_sequence_standard20,['domain','ligand','ligand_sequence']].to_dict('records'),
        primary_recommendation=decisions[0],new_model_fits=0,new_search_trials=0,heldout_outcome_files_read=0)
    dump(OUT/'result.json',result)
    mapping=pd.DataFrame([{k:r[k] for k in ['domain','parent_gene','actual_length','metadata_matches','aux_positions','helix','unmapped_alignment_tail']} for r in maps])
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>辅助功能数据可用性审计</title><style>body{{font:16px/1.7 system-ui;max-width:1280px;margin:40px auto;padding:0 24px;color:#18382d;background:#f7faf8}}table{{border-collapse:collapse;width:100%;font-size:13px;background:#fff}}th,td{{padding:9px;border-bottom:1px solid #dce6df;text-align:left;overflow-wrap:anywhere}}th{{background:#e7f1e9}}h2{{font-size:22px;margin-top:30px}}.note{{background:#e4f1e8;border-left:5px solid #337658;padding:20px}}.scroll{{overflow-x:auto}}</style>
<h1>配体、丰度、结合及重复测量：辅助数据审计</h1><p class="note">结论：具备开展受控辅助信息试验的数据基础，优先PSD95-PDZ2。18个开发文件覆盖4个结构域、2个母蛋白、6种结合条件和4种丰度条件。没有训练模型或改变当前评测。</p>
<h2>可配对的数据</h2><p>matched_AA为同一block内具有有限聚合fitness/sigma的共同AA变体，跨block只按身份取并集；common_block_nt_records为匹配的核苷酸记录数，不是生物重复数。不同密码子和文库block不会作为独立生物重复平均。</p><div class="scroll">{paired.to_html(index=False,border=0)}</div>
<h2>同一蛋白的多配体＋丰度交集</h2>{multi.to_html(index=False,border=0)}
<h2>序列和编号核对</h2>{mapping.to_html(index=False,border=0)}<p>四个结构域的同源九位点均定位到作者结构比对的82–90列。PDZ3实际输入与作者FASTA一致，为84个AA加终止符；元数据/结构比对多出末端E，差异不在九个扫描位点内。实际测量表作为变体编号参考，差异保留。2018完整实验构建体仍不能据此确认。</p>
<h2>与2018的变异身份重叠</h2>{overlap.to_html(index=False,border=0)}<p>同位点替换模式并不意味着完整构建体、测定或数值单位相同；必须以跨研究、跨配体的共同变异组保守控制泄漏。2026 PSD95-PDZ3复用2022数据，不能将两篇论文当作独立验证来源。</p>
<h2>准入决定</h2>{pd.DataFrame(decisions).to_html(index=False,border=0)}
<h2>最小接入方案</h2><p>先以PSD95-PDZ2的丰度和两种已知肽结合数据检验新增功能信息。在原训练规则下，外层留出的同源位点必须从所有辅助监督标签中排除。若把外部单突变读数作为公开参考提供给任务，必须另立信息增强设置，并向所有对照提供同样信息。未知母蛋白泛化需要将DLG4两个域一起留出；PDZK1同理。</p><p>使用测定特异的输出尺度与误差。丰度仅为约束折叠的表型代理，bindingPCA不是亲和力；不能直接相减当作结合能，也不使用作者在完整数据上拟合的MoCHI参数。先比较同信息访问的小型对照，再评估机制模型的额外价值。</p>
<h2>数据质量与隔离</h2><p>逐重复fitness相关范围为{result['descriptive_replicate_correlation_range'][0]:.3f}–{result['descriptive_replicate_correlation_range'][1]:.3f}，按文件去重后的单/双变体描述，不是统一的测量置信区间。每个文件的具体可用率、计数及相关见file_audit.csv。</p><p>独立核对{verification['catalog_rows_independently_checked']}条身份记录及最大输入计数代表选择，复核6组表型配对、4组同源位点映射和15组隔离计数；旧结果哈希不变。ERBIN/SHANK3/ZO1结果及作者拟合权重未读取。</p>
<p><a href="file_audit.csv">逐文件质量审计</a> · <a href="integration_plan.json">接入方案</a> · <a href="split_eligibility.csv">隔离后可用量</a> · <a href="verification.json">验证</a> · <a href="https://doi.org/10.1038/s41467-026-71005-x">来源论文</a> · <a href="https://github.com/lehner-lab/PDZ_homologs">作者代码</a></p></html>'''
    (OUT/'report.html').write_text(text)
    dump(OUT/'final_artifact_hashes.json',hashes([p for p in OUT.rglob('*') if p.is_file() and p.name!='final_artifact_hashes.json' and p.suffix!='.log']))
    print(paired.to_string(index=False));print(multi.to_string(index=False));print(overlap.to_string(index=False));print(json.dumps({k:result[k] for k in ['development_files','domains','parent_proteins','metadata_sequence_mismatches','ligand_sequence_unknown']},indent=2))


if __name__=='__main__':main()
