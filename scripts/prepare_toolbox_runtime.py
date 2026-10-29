"""Prepare the official SDK in an isolated copy; preserve the original vendor tree."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / '.local/science_toolbox_v1'


def main():
    runtime = PRIVATE / 'agh-runtime'
    PRIVATE.mkdir(parents=True, exist_ok=True)
    if not runtime.exists():
        # APFS clone avoids duplicating the vendored dependency data on this Mac.
        subprocess.run(['/bin/cp', '-cR', str(ROOT/'vendor/agnes-harness'), str(runtime)], check=True)
    restored = 0
    for _ in range(3):
        changed = 0
        for path in runtime.rglob('*'):
            if path.is_symlink() or not path.is_file() or 'node_modules' not in path.parts or path.stat().st_size >= 250:
                continue
            try:
                target = path.read_text()
            except UnicodeDecodeError:
                continue
            if '\n' not in target and target.startswith(('../', '.pnpm/')) and (path.parent/target).exists():
                path.unlink(); path.symlink_to(target); changed += 1
        restored += changed
        if not changed:
            break
    cli = PRIVATE/'node-tools/node_modules/tsx/dist/cli.mjs'
    if not cli.exists():
        subprocess.run(['npm', 'install', '--prefix', str(PRIVATE/'node-tools'), '--no-audit', '--no-fund',
                        'tsx@4.23.0', 'esbuild@0.28.0'], check=True)
    print(f'Isolated official SDK ready; restored {restored} dependency links; original vendor unchanged')


if __name__ == '__main__':
    main()
