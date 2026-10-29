"""Frozen diagnosis and grouped, budgeted development experiment."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from pdz2018_data import ROOT, RT, DOMAINS, dump, hashes, load_development
from pdz2018_policy import PDZPolicy
from pdz_background_model import training_data, ResidualModel, OnlinePredictor, site

OUT = ROOT/'experiments/pdz2018_background_v1'
METHODS = ['wt_rank','contact_seed0','additive_active','ridge_seq_active','ridge_struct_active','ridge_struct_uncertainty']


def preserved_hashes():
    expected = {}
    for folder in ['gb1_final_holdout_v1','pdz2018_point_dev_v1','pdz2018_uncertainty_v1']:
        p = ROOT/'experiments'/folder
        freeze = json.loads((p/'freeze.json').read_text())
        expected.update(freeze['hashes'])
        if (p/'setup_addendum.json').exists():
            expected.update(json.loads((p/'setup_addendum.json').read_text())['hashes'])
        name = 'final_artifact_hashes.json' if (p/'final_artifact_hashes.json').exists() else 'artifact_hashes.json'
        if (p/name).exists():
            for key,val in json.loads((p/name).read_text()).items():
                if isinstance(val,str):
                    target = ROOT/key if key.startswith(('experiments/','scripts/','data/')) else p/key
                    expected[str(target.relative_to(ROOT))] = val
    for path,digest in expected.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    return expected


def freeze():
    assert not (OUT/'freeze.json').exists(),'Never replace a frozen experiment'
    OUT.mkdir(parents=True,exist_ok=True)
    dump(OUT/'preserved_hashes.json',preserved_hashes())
    protocol = dict(version=OUT.name,scope='post-old-audit exploratory development; point energy only',
        domains=list(DOMAINS),sealed_unread=['shank3','zo1'],tasks=207,budget=32,initial_WT=8,batch=4,
        short_pool='retain and exhaust min(32,N)',target='epsilon_G = G(D+R)-G(D)-G(R); favorable negative; kcal/mol',
        endpoint='exp(-Gcombo/RT)>=0.5 WT',
        training='All valid canonical development double genotypes; each site fold excludes doubles touching held damage site on EITHER endpoint across ALL three domains. Public singles allowed. Extra historical double training data disclosed; not a singles-only comparison.',
        ridge_alpha=10.,scaling='train fold only; intercept unpenalized; no tuning',
        features='canonical endpoint position/native AA/substitution AA/substitution chemical group, single energies, volume change, P/G flags, group pair and scalar products; structure adds WT distance/6, exp(-(distance/6)^2) and proximity interactions',
        online='fixed RBF GP on public rescue features, unit prior variance kcal^2, regularization noise variance .09 kcal^2, NOT an estimated biological uncertainty; update on committed queried residuals only',
        acquisition='LCB predicted Gcombo minus one posterior SD; uncertainty arm greedy conditional variance; whole batch committed before label reveal',
        methods=METHODS,matched_MAE='all unqueried candidates after SAME initial8 labels; task-macro and domain metrics; additive zero and additive online controls',
        model_gate=dict(net_success_vs_WT_at_least=3,gained_damage_sites_at_least=2,MAE_reduction_vs_additive_zero_at_least=.05),
        selection='among learned methods maximize full207 successes; tie lower shared8 MAE then sequence simplicity; one fixed evaluation, no tuning',
        structure_support='shared8 structure MAE <= .95 * sequence MAE and structure LCB success >= sequence LCB success; conditional development evidence only',
        stage3='Only if model gate passes: same original identity-selected32 cohort, same chosen predictor/GP, FULL legal pool, initial8 and32 budget; Agnes against strongest learned fixed arm plus WT. No answer-enriched cohort.',
        agnes_gate=dict(net_core_success_vs_strongest_fixed_at_least=3,gained_damage_sites_at_least=2,all_queries_complete=True,no_unrelated_tool_execution=True),
        sealed_test='remain unread during all three development steps; if agent gate passes, freeze final candidate before any separately authorized heldout evaluation',
        interpretation='No claim of biological confidence, causal mechanism, independent validation or pure agent contribution from predictor-only gains',
        method_sources=['https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification','https://gaussianprocess.org/gpml/chapters/RW2.pdf'])
    dump(OUT/'protocol.json',protocol)
    paths = [ROOT/'scripts'/n for n in ['pdz_background_model.py','run_pdz_background.py','pdz2018_data.py','pdz2018_policy.py']]
    paths += [OUT/'protocol.json',OUT/'preserved_hashes.json',ROOT/'data/public_reference/residue_volumes.json',ROOT/'data/pdz2018/source/Salinas_Ranganathan_data.mat']
    paths += list((ROOT/'data/pdz2018/public').glob('*_geometry.npz'))
    dump(OUT/'freeze.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes(paths),boundary='parameters fixed before this new model fit/evaluation; development labels previously audited'))
    print('Frozen background model protocol',flush=True)


def diagnosis(tasks):
    details,summary = [],[]
    old = ROOT/'experiments/pdz2018_point_dev_v1'
    for task_id,t in tasks.items():
        wt = PDZPolicy(t,'wt_rank')
        true = [i for i,r in enumerate(t['rows']) if r['affinity']>=.5]
        if not true or set(true)&set(wt.order[:32]): continue
        contact_order = np.lexsort((wt.ids,-wt.wt,-wt.contact))
        distance = np.load(ROOT/f"data/pdz2018/public/{t['domain']}_geometry.npz")['distance']
        raw = np.array([distance[site(t['damage']),site(r['id'])] for r in t['rows']])
        raw_order = np.lexsort((wt.ids,raw))
        # Counterfactual menus under fixed seed0, not actual agent exposure.
        seed = np.random.SeedSequence([20261008,list(DOMAINS).index(t['domain']),site(t['damage'])+1,ord(t['damage'][-1]),0])
        policy = PDZPolicy(t,'no_feedback_contact',seed=seed)
        hypothetical = []
        while len(policy.seen)<policy.cap:
            menu = policy.menu()
            if len(policy.seen)>=8: hypothetical.append((len(policy.seen),menu))
            policy.update(menu['fixed_rule_proposal'])
        path = old/'queried_results'/f'{task_id}.json'
        actual = json.loads(path.read_text()) if path.exists() else None
        actual_ids = {q['rescue'] for q in actual['queries']} if actual else set()
        for i in true:
            r = t['rows'][i]
            menus = []
            if actual:
                for d in actual['decisions']:
                    menu = d.get('menu') or {}
                    candidates = menu.get('candidates',[])
                    if r['id'] in [c['id'] for c in candidates]: menus.append(d.get('budget_before',d.get('queries_before')))
            details.append(dict(task_id=task_id,domain=t['domain'],damage=t['damage'],rescue=r['id'],affinity=r['affinity'],
                epsilon_G_kcal=r['deltaG_kcal']+RT*(t['damage_log_affinity']+r['WT_log_affinity']),
                WT_rank=int(np.flatnonzero(wt.order==i)[0])+1,contact_rank=int(np.flatnonzero(contact_order==i)[0])+1,
                raw_distance_rank=int(np.flatnonzero(raw_order==i)[0])+1,distance_A=float(raw[i]),chemical_gate=bool(wt.contact[i]>0),
                fixed_menu_budgets=json.dumps([b for b,m in hypothetical if r['id'] in [c['id'] for c in m['candidates']]]),
                actual_Agnes_task=actual is not None,actual_menu_budgets=json.dumps(menus) if actual else None,
                actual_selected=r['id'] in actual_ids if actual else None))
        summary.append(dict(task_id=task_id,domain=t['domain'],damage=t['damage'],candidates=len(t['rows']),true_rescues=len(true),
            minimum_WT_rank=min(int(np.flatnonzero(wt.order==i)[0])+1 for i in true),
            contact_seed0_success=bool(set(true)&set(policy.seen)),actual_Agnes_task=actual is not None))
    assert len(summary)==18
    pd.DataFrame(details).to_csv(OUT/'missed_rescue_details.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'missed_backgrounds.csv',index=False)
    print('Diagnosed 18 WT misses; true rescue pairs:',len(details),flush=True)


def evaluate():
    assert not (OUT/'result.json').exists(),'Frozen output cannot be rerun/overwritten'
    f = json.loads((OUT/'freeze.json').read_text())
    assert all(hashes([ROOT/k])[k]==v for k,v in f['hashes'].items())
    tasks,records = load_development(),training_data()
    keys = [(r['domain'],r['left'],r['right']) for r in records]
    assert len(keys)==len(set(keys))
    diagnosis(tasks)
    pd.DataFrame(records).to_csv(OUT/'historical_training_genotypes.csv.gz',index=False,compression='gzip')
    models = {}
    audits = []
    for held in range(9):
        for structure in [False,True]:
            m = ResidualModel(records,held,structure)
            models[held,structure] = m;audits.append(m.audit)
        print('Trained site fold',held+1,flush=True)
    dump(OUT/'fold_audit.json',audits)
    runs,mae_rows,query_log,decisions = [],[],[],[]
    for task_id,t in tasks.items():
        rows = t['rows'];ids = np.array([r['id'] for r in rows])
        wt = PDZPolicy(t,'wt_rank');initial = list(map(int,wt.order[:8]));cap = min(32,len(rows))
        additive = -RT*(t['damage_log_affinity']+np.array([r['WT_log_affinity'] for r in rows]))
        epsilon = np.array([r['deltaG_kcal'] for r in rows])-additive
        means = {False:models[site(t['damage']),False].predict(t),True:models[site(t['damage']),True].predict(t)}
        unseen = [i for i in range(len(rows)) if i not in initial]
        mae = {}
        for name,structure,base in [('additive_online',False,np.zeros(len(rows))),('sequence',False,means[False]),('structure',True,means[True])]:
            predictor = OnlinePredictor(t,base,structure)
            predictor.update(initial,epsilon[initial]);pred,_ = predictor.posterior()
            mae[name] = float(np.mean(abs(pred[unseen]-epsilon[unseen])))
            mae_rows.append(dict(task_id=task_id,domain=t['domain'],damage_site=site(t['damage'])+1,model=name,n=len(unseen),MAE_kcal=mae[name]))
        mae['additive_zero'] = float(np.mean(abs(epsilon[unseen])))
        mae_rows.append(dict(task_id=task_id,domain=t['domain'],damage_site=site(t['damage'])+1,model='additive_zero',n=len(unseen),MAE_kcal=mae['additive_zero']))
        for method in METHODS:
            structure = method.startswith('ridge_struct')
            base = means[structure] if method.startswith('ridge_') else np.zeros(len(rows))
            predictor = OnlinePredictor(t,base,structure)
            seed = np.random.SeedSequence([20261008,list(DOMAINS).index(t['domain']),site(t['damage'])+1,ord(t['damage'][-1]),0])
            policy = PDZPolicy(t,'no_feedback_contact' if method=='contact_seed0' else 'wt_rank',seed=seed)
            seen = []
            while len(seen)<cap:
                size = min(4,cap-len(seen))
                before = len(seen)
                mean,cov = predictor.posterior()
                if before<8: batch = initial[before:before+size]
                elif method in ['wt_rank','contact_seed0']: batch = policy.proposal()
                else: batch = predictor.proposal(size,'variance' if method.endswith('uncertainty') else 'lcb')
                assert len(set(batch))==size and not set(batch)&set(seen)
                decisions.append(dict(task_id=task_id,method=method,budget_before=before,
                    selected=[str(ids[i]) for i in batch],predicted_residuals=[float(mean[i]) for i in batch]))
                # Whole batch is selected before the evaluator reveals any outcome.
                predictor.update(batch,epsilon[batch]);policy.update([str(ids[i]) for i in batch])
                for i in batch:
                    seen.append(i)
                    query_log.append(dict(task_id=task_id,method=method,query=len(seen),batch_before=before,rescue=ids[i],
                        affinity=rows[i]['affinity'],epsilon_G_kcal=float(epsilon[i]),prereveal_prediction=float(mean[i])))
            best = max(rows[i]['affinity'] for i in seen)
            runs.append(dict(task_id=task_id,domain=t['domain'],damage=t['damage'],damage_site=site(t['damage'])+1,
                method=method,success=best>=.5,best_affinity=best,queries=len(seen),point_reachable=any(r['affinity']>=.5 for r in rows),
                shared8_MAE_kcal=mae['structure' if structure else 'sequence'] if method.startswith('ridge_') else mae['additive_online']))
    runs = pd.DataFrame(runs);mae_frame = pd.DataFrame(mae_rows)
    runs.to_csv(OUT/'runs.csv',index=False);mae_frame.to_csv(OUT/'matched_MAE.csv',index=False)
    pd.DataFrame(query_log).to_csv(OUT/'query_log.csv.gz',index=False,compression='gzip')
    with gzip.open(OUT/'decisions.jsonl.gz','wt') as stream:
        for d in decisions: stream.write(json.dumps(d,allow_nan=False)+'\n')
    summary = []
    wt = runs.query("method=='wt_rank'").set_index('task_id')
    for method in METHODS:
        frame = runs[runs.method==method].set_index('task_id')
        gains = frame.success & ~wt.success;losses = ~frame.success & wt.success
        summary.append(dict(method=method,successes=int(frame.success.sum()),tasks=len(frame),success_rate=float(frame.success.mean()),
            gains=int(gains.sum()),losses=int(losses.sum()),net=int(gains.sum()-losses.sum()),
            gained_sites=int(frame.loc[gains,'damage_site'].nunique()),shared8_MAE_kcal=float(frame.shared8_MAE_kcal.mean())))
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    candidates = [s for s in summary if s['method'].startswith('ridge_')]
    chosen = sorted(candidates,key=lambda s:(-s['successes'],s['shared8_MAE_kcal'],s['method']!='ridge_seq_active',s['method']))[0]
    mae_by_model = mae_frame.groupby('model').MAE_kcal.mean().to_dict()
    checks = dict(net_success=chosen['net']>=3,multiple_gained_sites=chosen['gained_sites']>=2,
        prediction_MAE=chosen['shared8_MAE_kcal']<=.95*mae_by_model['additive_zero'])
    seq = next(s for s in summary if s['method']=='ridge_seq_active')
    struct = next(s for s in summary if s['method']=='ridge_struct_active')
    dump(OUT/'result.json',dict(summary=summary,matched_MAE=mae_by_model,selected_method=chosen['method'],
        gate_checks=checks,model_gate_passed=all(checks.values()),
        structure_supported=mae_by_model['structure']<=.95*mae_by_model['sequence'] and struct['successes']>=seq['successes'],
        historical_unique_genotypes=len(records),old_artifacts_unchanged=bool(preserved_hashes()),
        sealed_domains_loaded=False,stage3_status='eligible' if all(checks.values()) else 'skipped_per_frozen_gate',
        evidence='exploratory grouped development; changed historical-data access; point-only; no biological CI'))
    print(json.dumps(json.loads((OUT/'result.json').read_text()),indent=2),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='freeze': freeze()
    elif sys.argv[1]=='evaluate': evaluate()
    else: raise ValueError('Use freeze or evaluate')
