#!/usr/bin/env python3
"""Freeze the code a benchmark runs.

Copies the ``Ageas`` checkout from Ageas.share into ``<out>/code`` and links
the data folder next to it (the registry's ``../data`` paths). It checks that
the benchmark's data files match Ageas.share, and hashes the ones only the
local folder holds, such as Zheng68K. It then writes
``<out>/code/SNAPSHOT.json`` with the checkout's git state: commit, the
uncommitted diff of tracked files, and a hash of every untracked file, so
code that is not committed yet is recorded too. Jobs import the frozen copy,
so edits in Ageas.share during a multi-day run never reach it. An existing
snapshot is never replaced.

usage: snapshot.py --out <bench dir> [--share <Ageas.share>] [--data <data dir>]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

#: The folder holding the Ageas checkout this script sits in (``<share>/Ageas/revision/v2/snapshot.py``).
SHARE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#: The simulated datasets are covered by their md5 list (check the copy with ``md5sum -c`` first).
DATASETS = ['larry/larry_bench.h5ad', 'celltag/celltag_multi_rna.h5ad', 'celltag/atac_peaks_top50k.h5ad',
            'spatial/axolotl_d15.h5ad', 'simulation/MD5SUMS.txt']
#: Data the jobs read that Ageas.share does not hold (the registry notes where each came from):
#: hashed into the record, not compared.
LOCAL_ONLY = ['zheng68k/LiuBIB_5_Zheng68K(PBMC)_FACS_label.h5ad']


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def git(repo, *args):
    """Output of a git command in ``repo``, or ``None`` if it fails (e.g. no commits yet)."""
    proc = subprocess.run(['git', '-c', 'safe.directory=*', '-C', repo, *args], capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else None


def repo_state(repo, diff_path) -> dict:
    head = git(repo, 'rev-parse', 'HEAD')
    status = git(repo, 'status', '--porcelain') or ''
    diff = git(repo, 'diff', 'HEAD') if head else None
    if diff:
        with open(diff_path, 'w', encoding='utf-8') as fh:
            fh.write(diff)
    return {
        'commit': head.strip() if head else None,
        'branch': (git(repo, 'rev-parse', '--abbrev-ref', 'HEAD') or '').strip() or None,
        'dirty': bool(status.strip()),
        'uncommitted_diff': os.path.basename(diff_path) if diff else None,
        'untracked': [line[3:] for line in status.splitlines() if line.startswith('??')],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, help='Benchmark folder; the copy goes to <out>/code.')
    parser.add_argument('--share', default=SHARE,
                        help='Folder holding the Ageas checkout and data/ (default: the one this script sits in).')
    parser.add_argument('--data', default='/home/jack/ageas_work/data',
                        help='Data folder the jobs read (a local copy of Ageas.share/data).')
    args = parser.parse_args(argv)

    code = os.path.join(args.out, 'code')
    if os.path.exists(code):
        sys.exit(f'{code} exists; a benchmark keeps the code it started with.')
    data_hashes = {}
    for name in DATASETS:
        local, shared = sha256(os.path.join(args.data, name)), sha256(os.path.join(args.share, 'data', name))
        if local != shared:
            sys.exit(f'{name} in {args.data} differs from Ageas.share; refresh the local copy first.')
        data_hashes[name] = local
    for name in LOCAL_ONLY:
        data_hashes[name] = sha256(os.path.join(args.data, name))

    os.makedirs(code)
    subprocess.run(['rsync', '-a', '--exclude', '__pycache__', '--exclude', '.pytest_cache', '--exclude', '.claude',
                    '--exclude', 'revision/results', '--exclude', 'docs',
                    os.path.join(args.share, 'Ageas') + '/', os.path.join(code, 'Ageas') + '/'], check=True)
    os.symlink(args.data, os.path.join(code, 'data'))

    repo = os.path.join(code, 'Ageas')
    untracked = sorted((git(repo, 'ls-files', '--others', '--exclude-standard') or '').splitlines())
    record = {
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source': args.share,
        'ageas': dict(repo_state(repo, os.path.join(code, 'ageas_uncommitted.diff')),
                      untracked_sha256={f: sha256(os.path.join(repo, f)) for f in untracked}),
        'data': {'folder': args.data, 'sha256': data_hashes},
    }
    with open(os.path.join(code, 'SNAPSHOT.json'), 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=1)
    print(json.dumps({k: v for k, v in record.items() if k != 'ageas'}, indent=1))
    print(f"ageas: commit {record['ageas']['commit']}, dirty {record['ageas']['dirty']}, "
          f"{len(untracked)} untracked files hashed")


if __name__ == '__main__':
    main()
