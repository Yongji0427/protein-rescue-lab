"""Task-scoped scientific tools. Only the trusted evaluator owns hidden labels."""
import copy
import hashlib
import json
import math
from pathlib import Path
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from toolbox_python import run_python


VERSION = 'science_toolbox_v1'
GROUPS = dict(hydrophobic='AVLIM', aromatic='FWY', polar='STNQ', basic='KRH', acidic='DE', special='CGP')
READ_TOOLS = {'science_state', 'science_candidates', 'science_sequence', 'science_structure', 'science_controls'}
TOOLS = READ_TOOLS | {'science_predict', 'science_python', 'science_record', 'science_query', 'science_finish'}


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.chmod(0o600); tmp.replace(path)


class ToolboxTask:
    def __init__(self, path, spec):
        self.path = Path(path); self.spec = spec
        self.candidates = {r['id']: r for r in spec['candidates']}
        self.labels = spec['labels']
        assert self.candidates.keys() == self.labels.keys()
        for candidate in self.candidates:
            self.legal(candidate)
        fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state['fingerprint'] != fingerprint:
                raise ValueError('Existing task inputs differ; use a new version, never reset')
        else:
            self.state = dict(task_id=spec['task_id'], fingerprint=fingerprint,
                task_token=secrets.token_hex(16), session_key=VERSION + '-' + spec['task_id'],
                observed=[], decisions=[], hypotheses=[], analyses=[], predictions=[], conclusion=None,
                compute_used=0, analysis_used=0, requests={}, events=[])
            for candidate in spec['initial_ids']:
                assert candidate not in {r['id'] for r in self.state['observed']}
                self.state['observed'].append(dict(id=candidate, **self.labels[candidate], source='shared_initial_WT'))
            assert len(self.state['observed']) <= spec['query_budget']
            save(self.path, self.state)

    def legal(self, candidate):
        import re
        match = re.fullmatch(r'([ARNDCQEGHILKMFPSTWYV])(\d+)([ARNDCQEGHILKMFPSTWYV])', candidate)
        if not match:
            raise ValueError('Invalid mutation identity')
        native, position, mutant = match.groups(); position = int(position)
        sequence = self.spec['scanned_sequence']
        if not 1 <= position <= len(sequence) or sequence[position-1] != native or native == mutant:
            raise ValueError('Mutation does not match reference sequence')
        if position == int(self.spec['damage'][1:-1]):
            raise ValueError('Original damage site must remain locked')
        return position

    def visible(self):
        return dict(task_id=self.state['task_id'], kind=self.spec['kind'], domain=self.spec['domain'],
            damage=self.spec['damage'], damage_affinity=self.spec['damage_affinity'],
            scanned_sequence=self.spec['scanned_sequence'], sequence_scope='nine assayed sites, not full protein sequence',
            volume_feature='AAindex FAUJ880103 normalized van der Waals index; not volume in cubic angstroms',
            structure=self.spec['structure'], candidates=list(self.candidates.values()),
            observed=copy.deepcopy(self.state['observed']), hypotheses=copy.deepcopy(self.state['hypotheses']),
            query_budget=self.spec['query_budget'], queries_used=len(self.state['observed']),
            recovery_threshold=.5, measurement=('synthetic feedback for engineering only' if self.spec['kind'].startswith('synthetic')
                else 'historical WT-relative association affinity; point only'),
            evidence_boundary=self.spec['evidence_boundary'])

    def public(self):
        data = self.visible()
        # Full public features remain available through inspection tools and Python.
        # Repeating the entire pool/structure after every query wastes model context.
        data['candidate_count'] = len(data.pop('candidates'))
        structure = data.pop('structure')
        data['structure_reference'] = {k:structure[k] for k in ('pdb', 'chain', 'residue_numbers', 'description') if k in structure}
        data.update(task_token=self.state['task_token'], queries_remaining=self.spec['query_budget'] - len(self.state['observed']),
            compute_used=self.state['compute_used'], compute_budget=80,
            python_used=self.state['analysis_used'], python_budget=12,
            conclusion=self.state['conclusion'], finished=self.state['conclusion'] is not None)
        best = max(self.state['observed'], key=lambda r:r['affinity'], default=None)
        data['best_observed'] = best
        data['measured_point_recovery_found'] = bool(best and best['affinity'] >= .5)
        data['best_observed_controls'] = self.controls(best['id']) if best else None
        return data

    def controls(self, candidate):
        self.legal(candidate)
        if candidate not in self.candidates:
            raise ValueError('Candidate outside legal measured pool')
        observed = next((r for r in self.state['observed'] if r['id'] == candidate), None)
        if observed is None:
            raise ValueError('Control check requires an already queried combination')
        single = self.candidates[candidate]['WT_affinity']
        damage = self.spec['damage_affinity']; combo = observed['affinity']
        return dict(rescue_id=candidate, genotype=self.spec['damage'] + ':' + candidate,
            WT=1., damage=damage, rescue_single=single, damage_plus_rescue=combo,
            damage_retained=True, additional_substitutions=1, point_recovery=combo >= .5,
            log_nonadditivity=math.log(combo) - math.log(damage) - math.log(single),
            evidence='point measurement supports functional comparison; nonadditivity does not establish mechanism')

    def features(self):
        rows = list(self.candidates.values()); groups = list(GROUPS)
        return rows, np.array([[r['WT_log_affinity'], r['distance_angstrom']/6,
            r['volume_change']/100, float(r['mutant'] == 'P'), float(r['mutant'] == 'G')]
            + [float(r['group'] == g) for g in groups] for r in rows])

    def predict(self, method):
        if method not in ('additive', 'ridge'):
            raise ValueError('Unknown prediction method')
        rows, x = self.features(); additive = np.array([math.log(self.spec['damage_affinity']) + r['WT_log_affinity'] for r in rows])
        seen = {r['id']: r for r in self.state['observed']}
        prediction = additive.copy(); diagnostic = None
        if method == 'ridge':
            if len(seen) < 4:
                raise ValueError('Ridge requires at least four observed combinations')
            idx = np.array([i for i, r in enumerate(rows) if r['id'] in seen])
            center = x[idx].mean(0); scale = x[idx].std(0); scale[scale < 1e-8] = 1.
            z = np.c_[np.ones(len(x)), (x-center)/scale]
            y = np.array([math.log(seen[rows[i]['id']]['affinity']) - additive[i] for i in idx])
            penalty = np.eye(z.shape[1]) * 10.; penalty[0, 0] = 1e-6
            fit = np.linalg.solve(z[idx].T @ z[idx] + penalty, z[idx].T @ y)
            prediction += z @ fit
            hat = np.sum((z[idx] @ np.linalg.inv(z[idx].T @ z[idx] + penalty)) * z[idx], axis=1)
            loo = (y - z[idx] @ fit) / np.maximum(1.-hat, 1e-8)
            diagnostic = dict(observed_combinations=len(idx), leave_one_out_residual_MAE=float(np.abs(loo).mean()),
                warning='small observed sample; diagnostic error is not independent validation or biological uncertainty')
        ranked = sorted([dict(id=r['id'], predicted_log_affinity=float(prediction[i]),
            predicted_affinity=float(math.exp(float(np.clip(prediction[i], -50, 50)))))
            for i, r in enumerate(rows) if r['id'] not in seen], key=lambda r: (-r['predicted_log_affinity'], r['id']))
        result = dict(method=method, ranked=ranked[:20], unqueried_candidates=len(ranked),
            ranking_scope='top20; full pool available to science_candidates and science_python', diagnostic=diagnostic,
            fitted_ids=sorted(seen) if method == 'ridge' else [], evidence='computed prediction; not experimental measurement')
        self.state['predictions'].append(result)
        return result

    def dispatch(self, action, args):
        if action == 'science_state':
            return self.public()
        if action == 'science_sequence':
            return dict(scanned_sequence=self.spec['scanned_sequence'], damage=self.spec['damage'],
                reference_chain_sequence=self.spec['structure'].get('reference_chain_sequence'),
                reference_sequence_scope=self.spec['structure'].get('reference_sequence_scope', 'synthetic scanned sites only'),
                volume_feature='AAindex FAUJ880103 normalized volume index; not cubic angstroms',
                candidates=list(self.candidates.values()), evidence='public singles, sequence and residue properties only')
        if action == 'science_structure':
            return dict(**self.spec['structure'], evidence='reference WT geometry; no mutant structural or mechanism validation')
        if action == 'science_candidates':
            rows = list(self.candidates.values())
            if 'site' in args:
                if type(args['site']) is not int or not 1 <= args['site'] <= 9:
                    raise ValueError('Candidate site must be 1..9')
                rows = [r for r in rows if r['site'] == args['site']]
            if 'group' in args:
                if args['group'] not in GROUPS:
                    raise ValueError('Unknown amino-acid group')
                rows = [r for r in rows if r['group'] == args['group']]
            rank = args.get('rank', 'wt')
            if rank not in ('wt', 'distance', 'id'):
                raise ValueError('Unknown public ranking')
            key = {'wt': lambda r: (-r['WT_log_affinity'], r['id']),
                   'distance': lambda r: (r['distance_angstrom'], r['id']), 'id': lambda r: r['id']}[rank]
            seen = {r['id'] for r in self.state['observed']}
            return dict(candidates=[dict(r, queried=r['id'] in seen) for r in sorted(rows, key=key)],
                full_pool_size=len(self.candidates), evidence='pool coverage and public features; no unqueried outcomes')
        if action == 'science_controls':
            return self.controls(args['rescue_id'])
        if action == 'science_predict':
            return self.predict(args.get('method', 'additive'))
        if action == 'science_python':
            purpose = args.get('purpose', '')
            if not isinstance(purpose, str) or not 1 <= len(purpose) <= 500:
                raise ValueError('State the analysis purpose')
            if self.state['analysis_used'] >= 12:
                raise ValueError('Python analysis budget exhausted')
            self.state['analysis_used'] += 1
            save(self.path, self.state)  # An attempt remains charged across failure/restart.
            try:
                result = run_python(args['code'], self.visible())
            except (ValueError, RuntimeError) as error:
                result = dict(ok=False, error=str(error), sandbox='execution refused')
            self.state['analyses'].append(dict(purpose=purpose, code=args['code'], queries_used=len(self.state['observed']), result=result))
            return result
        if action == 'science_record':
            status = args.get('status'); claim = args.get('claim'); evidence = args.get('evidence_ids')
            if status not in ('proposed', 'supported', 'contradicted', 'inconclusive') or not isinstance(claim, str) or not 1 <= len(claim) <= 1000:
                raise ValueError('Provide a bounded claim and valid status')
            seen = {r['id'] for r in self.state['observed']}
            if not isinstance(evidence, list) or len(evidence) > 32 or any(r not in seen for r in evidence):
                raise ValueError('Evidence IDs must reference already observed combinations')
            hid = args.get('hypothesis_id') or 'H' + str(len(self.state['hypotheses'])+1)
            if len(hid) > 64 or ('hypothesis_id' in args and hid not in {r['hypothesis_id'] for r in self.state['hypotheses']}):
                raise ValueError('Unknown hypothesis revision')
            note = dict(hypothesis_id=hid, claim=claim, status=status, evidence_ids=evidence,
                next_test=str(args.get('next_test', ''))[:1000], queries_used=len(seen))
            self.state['hypotheses'].append(note)
            return dict(recorded=note, evidence='agent hypothesis; not independently validated mechanism')
        if action == 'science_query':
            ids = args.get('rescue_ids'); predictions = args.get('predicted_affinities')
            if not isinstance(ids, list) or not 1 <= len(ids) <= 4 or any(not isinstance(r, str) for r in ids) or len(set(ids)) != len(ids):
                raise ValueError('Commit 1..4 unique candidate IDs')
            if not isinstance(predictions, list) or len(predictions) != len(ids) or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in predictions):
                raise ValueError('Provide one finite nonnegative prereveal prediction per candidate')
            for candidate in ids:
                self.legal(candidate)
                if candidate not in self.candidates:
                    raise ValueError('Candidate outside measured legal pool')
            previous = next((d for d in self.state['decisions'] if d['selected'] == ids), None)
            if previous:
                return dict(state=self.public(), new_queries_charged=0, idempotent_replay=True)
            seen = {r['id'] for r in self.state['observed']}
            if seen & set(ids):
                raise ValueError('Partial duplicate or already observed candidate')
            if len(seen) + len(ids) > self.spec['query_budget']:
                raise ValueError('Experimental query budget exhausted')
            rationale = args.get('rationale')
            if not isinstance(rationale, str) or not 1 <= len(rationale) <= 1000:
                raise ValueError('Provide the prereveal test rationale')
            self.state['decisions'].append(dict(budget_before=len(seen), selected=ids,
                predicted_affinities=predictions, rationale=rationale))
            for candidate, predicted in zip(ids, predictions):
                self.state['observed'].append(dict(id=candidate, **self.labels[candidate],
                    prereveal_prediction=predicted, source='native_science_query'))
            return dict(state=self.public(), new_queries_charged=len(ids), idempotent_replay=False)
        if action == 'science_finish':
            outcome = args.get('outcome'); summary = args.get('summary'); unresolved = args.get('unresolved')
            if outcome not in ('measured_rescue', 'budget_exhausted', 'insufficient_evidence') or not isinstance(summary, str) or not 1 <= len(summary) <= 2000 or not isinstance(unresolved, str) or len(unresolved) > 2000:
                raise ValueError('Provide a bounded conclusion and unresolved questions')
            best = max(self.state['observed'], key=lambda r:r['affinity'], default=None)
            if outcome != 'measured_rescue' and best is not None and best['affinity'] >= .5:
                raise ValueError('A qualifying measured rescue already exists: ' + best['id'] +
                    '. Check its four controls and finish with measured_rescue; mechanism can remain unresolved.')
            controls = None
            if outcome == 'measured_rescue':
                controls = self.controls(args.get('rescue_id', ''))
                if not controls['point_recovery']:
                    raise ValueError('Success requires measured point recovery >=0.5WT')
            if outcome == 'budget_exhausted' and len(self.state['observed']) < self.spec['query_budget']:
                raise ValueError('Experimental budget has not been exhausted')
            if controls is None and best is not None:
                controls = self.controls(best['id'])
            self.state['conclusion'] = dict(outcome=outcome, summary=summary, unresolved=unresolved,
                controls=controls, best_observed=best, evidence_boundary=self.spec['evidence_boundary'])
            return dict(conclusion=self.state['conclusion'], finished=True)
        raise ValueError('Unknown scientific tool')

    def handle(self, payload):
        if payload.get('task_token') != self.state['task_token'] or payload.get('session_key') != self.state['session_key']:
            raise ValueError('Task token is bound to its AGH session')
        action = payload.get('action')
        if action not in TOOLS:
            raise ValueError('Tool outside scientific toolbox')
        if action in READ_TOOLS:
            return self.dispatch(action, payload)
        rid = payload.get('request_id')
        if not isinstance(rid, str) or len(rid) != 64 or any(c not in '0123456789abcdef' for c in rid):
            raise ValueError('Invalid trusted request ID')
        clean = {k: v for k, v in payload.items() if k not in ('task_token', 'session_key', 'request_id')}
        signature = hashlib.sha256(json.dumps(clean, sort_keys=True, allow_nan=False).encode()).hexdigest()
        prior = self.state['requests'].get(rid)
        if prior:
            if prior['signature'] != signature:
                raise ValueError('Request ID cannot change its payload')
            result = dict(prior['result'], request_replayed=True)
            if action == 'science_query':
                result['new_queries_charged'] = 0
            return result
        if self.state['conclusion'] is not None:
            raise ValueError('Task finished; use state or read-only evidence tools')
        if self.state['compute_used'] >= 80 and action != 'science_finish':
            raise ValueError('Scientific computation budget exhausted')
        before = copy.deepcopy(self.state)
        try:
            result = self.dispatch(action, payload)
        except Exception:
            # Failed Python runs are retained/charged; invalid biological requests are atomic.
            if action != 'science_python':
                self.state = before
            save(self.path, self.state)
            raise
        if action != 'science_finish' and not result.get('idempotent_replay', False):
            self.state['compute_used'] += 1
        self.state['events'].append(dict(action=action, queries_used=len(self.state['observed']), request_id=rid))
        self.state['requests'][rid] = dict(signature=signature, result=result)
        save(self.path, self.state)
        return result


def serve(tasks, endpoint):
    lock = threading.Lock(); by_token = {t.state['task_token']: t for t in tasks}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            try:
                if self.path != '/toolbox':
                    raise ValueError('Unknown toolbox endpoint')
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 32000:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(size))
                with lock:
                    task = by_token.get(payload.get('task_token'))
                    if task is None:
                        raise ValueError('Unknown task token')
                    value = task.handle(payload)
                status = 200
            except (ValueError, TypeError, KeyError, RuntimeError) as error:
                value = dict(error=str(error), new_queries_charged=0); status = 400
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded))); self.end_headers()
            try:
                self.wfile.write(encoded)
            except BrokenPipeError:
                pass  # State was committed; same request recovers it without new charges.
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    save(endpoint, dict(url=f'http://127.0.0.1:{server.server_port}/toolbox'))
    print(json.dumps(dict(ready=True, tasks=len(tasks), port=server.server_port)), flush=True)
    server.serve_forever()
