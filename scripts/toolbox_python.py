"""Execute agent analysis with OS-enforced read/write/network restrictions on macOS."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


WORKER = '''import contextlib, io, json, resource, traceback
resource.setrlimit(resource.RLIMIT_CPU, (6, 7))
resource.setrlimit(resource.RLIMIT_FSIZE, (20000, 20000))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
import numpy as np
import pandas as pd
payload = json.load(__import__('sys').stdin)
class LimitedOutput(io.StringIO):
    def write(self, value):
        if self.tell() + len(value) > 12000:
            raise ValueError('Analysis output limit exceeded')
        return super().write(value)
output = LimitedOutput()
scope = dict(data=payload['data'], np=np, pd=pd)
try:
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        exec(compile(payload['code'], '<agent-analysis>', 'exec'), scope)
    result = scope.get('result')
    if result is None and not output.getvalue().strip():
        raise ValueError('Set result or print an analysis result; empty output is not an analysis artifact')
    encoded = json.dumps(dict(ok=True, stdout=output.getvalue(), result=result), allow_nan=False)
    if len(encoded) > 16000:
        raise ValueError('Analysis result limit exceeded')
    print(encoded)
except BaseException as error:
    print(json.dumps(dict(ok=False, stdout=output.getvalue()[:12000], error=type(error).__name__ + ': ' + str(error)[:400])))
'''


def profile():
    if sys.platform != 'darwin' or not Path('/usr/bin/sandbox-exec').exists():
        raise RuntimeError('OS analysis sandbox unavailable; execution refused')
    # System libraries may be read. User data, temp state, other mounts, writes,
    # networking and subprocess creation are denied independently of Python code.
    blocked = ['/Users', '/Volumes', '/private/tmp', '/tmp', '/private/var/folders',
               '/var/folders', '/private/var/root', '/var/root', '/private/etc', '/etc']
    return '\n'.join(['(version 1)', '(deny default)', '(allow process*)',
        '(deny process-fork)', '(allow sysctl-read)', '(allow file-read*)',
        '(deny file-read* ' + ' '.join('(subpath ' + json.dumps(p) + ')' for p in blocked) + ')',
        '(deny network*)'])


def run_python(code, data, timeout=10):
    if not isinstance(code, str) or not code.strip() or len(code.encode()) > 12000:
        raise ValueError('Provide 1..12000 bytes of Python')
    policy = profile()
    # No parent environment, credentials, task token, endpoint or project sys.path.
    environment = dict(PATH='/usr/bin:/bin', HOME='/nonexistent',
        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        child = subprocess.Popen(['/usr/bin/sandbox-exec', '-p', policy,
            str(Path(sys.executable).resolve()), '-I', '-B', '-c', WORKER],
            stdin=subprocess.PIPE, stdout=output, stderr=errors, env=environment,
            cwd='/', start_new_session=True)
        child.stdin.write(json.dumps(dict(code=code, data=data), allow_nan=False).encode())
        child.stdin.close()
        deadline = time.monotonic() + timeout
        failure = None
        while child.poll() is None:
            if time.monotonic() >= deadline:
                failure = 'Analysis wall-time limit exceeded'
                break
            usage = subprocess.run(['/bin/ps', '-o', 'rss=', '-p', str(child.pid)],
                capture_output=True, text=True, timeout=2).stdout.strip()
            if usage.isdigit() and int(usage) > 512 * 1024:
                failure = 'Analysis memory limit exceeded (512 MiB RSS)'
                break
            try:
                child.wait(timeout=.1)
            except subprocess.TimeoutExpired:
                pass
        if failure:
            os.killpg(child.pid, signal.SIGKILL); child.wait()
            return dict(ok=False, error=failure, sandbox='macOS Seatbelt')
        output.seek(0); raw = output.read(16001)
        if child.returncode != 0 or len(raw) > 16000:
            return dict(ok=False, error='Analysis process failed or exceeded resource/output limits',
                        exit_code=child.returncode, sandbox='macOS Seatbelt')
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return dict(ok=False, error='Analysis must return one finite JSON result', sandbox='macOS Seatbelt')
        return dict(**value, sandbox='macOS Seatbelt', evidence='analysis of visible data; not a measurement')
