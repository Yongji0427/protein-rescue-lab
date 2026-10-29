"""Safety, scientific provenance and recovery checks on synthetic data only."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from prepare_science_toolbox import ROOT, OUT, synthetic_spec
from science_toolbox import ToolboxTask, save
from toolbox_python import run_python


class ToolboxTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / 'state.json'
        self.spec = synthetic_spec(); self.task = ToolboxTask(self.path, self.spec)
        self.counter = 0

    def tearDown(self):
        self.folder.cleanup()

    def call(self, action, **args):
        self.counter += 1
        return self.task.handle(dict(action=action, task_token=self.task.state['task_token'],
            session_key=self.task.state['session_key'], request_id=hashlib.sha256(str(self.counter).encode()).hexdigest(), **args))

    def unqueried(self):
        seen = {r['id'] for r in self.task.state['observed']}
        return [r for r in self.task.candidates if r not in seen]

    def test_visible_data_excludes_hidden_labels_and_tokens(self):
        visible = self.task.visible()
        self.assertNotIn('task_token', visible)
        self.assertNotIn('labels', visible)
        self.assertEqual(len(visible['observed']), 8)
        self.assertTrue(all('affinity' not in r and 'deltaG_kcal' not in r for r in visible['candidates']))
        public = self.call('science_state')
        self.assertNotIn('candidates', public)
        self.assertEqual(public['candidate_count'], len(visible['candidates']))
        self.assertEqual(len(self.call('science_candidates')['candidates']), len(visible['candidates']))

    def test_auth_illegal_and_partial_duplicate_requests_are_atomic(self):
        payload = dict(action='science_query', task_token=self.task.state['task_token'], session_key=self.task.state['session_key'],
            request_id='a'*64, rescue_ids=self.unqueried()[:4], predicted_affinities=[.1]*4, rationale='test')
        for edit in [dict(session_key='other'), dict(task_token='0'*32), dict(rescue_ids=['A4V']+payload['rescue_ids'][1:]),
                     dict(rescue_ids=['SHANK3_LABEL']+payload['rescue_ids'][1:]),
                     dict(rescue_ids=[self.task.state['observed'][0]['id']]+payload['rescue_ids'][1:]),
                     dict(predicted_affinities=[float('nan')]*4)]:
            before = copy.deepcopy(self.task.state)
            with self.assertRaises(ValueError):
                self.task.handle(dict(payload, **edit))
            self.assertEqual(before, self.task.state)

    def test_lost_response_restart_and_prereveal_prediction(self):
        ids = self.unqueried()[:4]
        p = dict(action='science_query', task_token=self.task.state['task_token'], session_key=self.task.state['session_key'],
            request_id='a'*64, rescue_ids=ids, predicted_affinities=[.12]*4, rationale='before reveal')
        self.assertEqual(self.task.handle(p)['new_queries_charged'], 4)
        self.assertEqual(self.task.handle(p)['new_queries_charged'], 0)
        self.task = ToolboxTask(self.path, self.spec)
        self.assertEqual(self.task.handle(dict(p, request_id='b'*64, predicted_affinities=[99]*4))['new_queries_charged'], 0)
        self.assertEqual(self.task.state['decisions'][0]['predicted_affinities'], [.12]*4)
        self.assertEqual(len(self.task.state['observed']), 12)
        with self.assertRaises(ValueError):
            self.task.handle(dict(p, rescue_ids=self.unqueried()[:4]))

    def test_exact_query_cap_and_finish_lock(self):
        spec = synthetic_spec()
        spec['labels'] = {k:dict(affinity=.1) for k in spec['labels']}
        self.task = ToolboxTask(Path(self.folder.name)/'no_rescues.json', spec)
        for _ in range(6):
            self.call('science_query', rescue_ids=self.unqueried()[:4], predicted_affinities=[.1]*4, rationale='test')
        self.assertEqual(self.call('science_state')['queries_remaining'], 0)
        with self.assertRaises(ValueError):
            self.call('science_query', rescue_ids=self.unqueried()[:1], predicted_affinities=[.1], rationale='excess')
        self.call('science_finish', outcome='budget_exhausted', summary='budget exhausted', unresolved='mechanism unknown')
        with self.assertRaises(ValueError):
            self.call('science_record', claim='after finish', status='proposed', evidence_ids=[], next_test='none')
        self.assertTrue(self.call('science_state')['finished'])

    def test_finish_cannot_ignore_shared_initial_rescue(self):
        self.assertTrue(self.call('science_state')['measured_point_recovery_found'])
        before = copy.deepcopy(self.task.state)
        with self.assertRaisesRegex(ValueError, 'qualifying measured rescue already exists'):
            self.call('science_finish', outcome='insufficient_evidence', summary='no rescue', unresolved='x')
        self.assertEqual(before, self.task.state)
        best = self.call('science_state')['best_observed']['id']
        result = self.call('science_finish', outcome='measured_rescue', rescue_id=best,
            summary='initial measured rescue retained', unresolved='mechanism unknown')
        self.assertTrue(result['finished'])

    def test_predictions_use_only_observed_labels(self):
        original = self.call('science_predict', method='ridge')
        for r in self.unqueried():
            self.task.labels[r] = dict(affinity=9999.)
        modified = self.call('science_predict', method='ridge')
        self.assertEqual(original, modified)
        self.assertEqual(set(original['fitted_ids']), {r['id'] for r in self.task.state['observed']})

    def test_hypotheses_require_visible_evidence_and_revisions_append(self):
        with self.assertRaises(ValueError):
            self.call('science_record', claim='bad', status='supported', evidence_ids=self.unqueried()[:1], next_test='x')
        h = self.call('science_record', claim='local effect', status='proposed', evidence_ids=[], next_test='query local candidates')
        self.call('science_record', hypothesis_id=h['recorded']['hypothesis_id'], claim='still uncertain',
            status='inconclusive', evidence_ids=[self.task.state['observed'][0]['id']], next_test='compare distant candidates')
        self.assertEqual(len(self.task.state['hypotheses']), 2)

    def test_four_controls_and_no_unmeasured_success(self):
        with self.assertRaises(ValueError):
            self.call('science_finish', outcome='measured_rescue', rescue_id=self.unqueried()[0], summary='bad', unresolved='x')
        rescue = next(k for k in self.unqueried() if self.task.labels[k]['affinity'] >= .5)
        self.call('science_query', rescue_ids=[rescue], predicted_affinities=[.6], rationale='test')
        controls = self.call('science_controls', rescue_id=rescue)
        self.assertTrue(controls['damage_retained'] and controls['point_recovery'])
        self.assertEqual(controls['additional_substitutions'], 1)
        self.assertTrue(self.call('science_finish', outcome='measured_rescue', rescue_id=rescue,
            summary='synthetic point success', unresolved='no biological claim')['finished'])

    def test_python_custom_ridge_and_stored_analysis(self):
        result = self.call('science_python', purpose='custom observed-only regression', code='''
rows = data['candidates']; observed = {r['id']: r['affinity'] for r in data['observed']}
x = np.array([[1., r['WT_log_affinity'], r['volume_change']/100] for r in rows])
indices = [i for i, r in enumerate(rows) if r['id'] in observed]
y = np.array([np.log(observed[rows[i]['id']]) for i in indices])
coef = np.linalg.solve(x[indices].T @ x[indices] + np.eye(3), x[indices].T @ y)
result = {'coefficients': coef.tolist(), 'observed_n': len(y)}
''')
        self.assertTrue(result['ok']); self.assertEqual(result['result']['observed_n'], 8)
        self.assertEqual(len(self.task.state['analyses']), 1)

    def test_python_budget_counts_failures(self):
        self.task.state['analysis_used'] = 11
        self.assertFalse(self.call('science_python', purpose='failure retained', code='raise ValueError("diagnostic")')['ok'])
        self.assertEqual(self.task.state['analysis_used'], 12)
        with self.assertRaises(ValueError):
            self.call('science_python', purpose='excess', code='result=1')


class SandboxTests(unittest.TestCase):
    def test_host_data_network_writes_and_fork_are_denied(self):
        code = '''
import os, socket, subprocess
from pathlib import Path
checks = {}
for name, operation in [
 ('host_read', lambda: Path('/etc/passwd').read_text()),
 ('tmp_read', lambda: Path('/tmp/protein-rescue-agh-20261008/data/daemon/owner.json').read_text()),
 ('write', lambda: Path('/tmp/toolbox-forbidden-write').write_text('x')),
 ('network', lambda: socket.create_connection(('127.0.0.1', 80), timeout=1)),
 ('subprocess', lambda: subprocess.run(['/bin/echo', 'x'], capture_output=True, check=True))]:
    try:
        operation(); checks[name] = False
    except PermissionError:
        checks[name] = True
result = {'blocked': checks, 'credential_env_present': any(k.startswith('AGNES') for k in os.environ),
          'task_token_present': 'task_token' in data}
'''
        value = run_python(code, {'observed': []})
        self.assertTrue(value['ok'], value)
        self.assertTrue(all(value['result']['blocked'].values()), value)
        self.assertFalse(value['result']['credential_env_present'])
        self.assertFalse(value['result']['task_token_present'])

    def test_timeout_and_output_cap(self):
        self.assertFalse(run_python('while True: pass', {}, timeout=1)['ok'])
        self.assertFalse(run_python('print("x"*13000)', {})['ok'])
        self.assertFalse(run_python('result=float("nan")', {})['ok'])
        self.assertFalse(run_python('x=1', {})['ok'])


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    save(OUT / 'offline_verification.json', dict(status='passed' if result.wasSuccessful() else 'failed',
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        data='synthetic only', sandbox='macOS Seatbelt + CPU/wall/RSS/output supervision',
        scientific_performance_claim=False))
    raise SystemExit(not result.wasSuccessful())
