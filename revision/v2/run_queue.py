#!/usr/bin/env python3
"""Run benchmark jobs, a few at a time, without starving other work on the machine.

A job is one (dataset, seed, method, fold), run by ``run_job.py`` in its own
process with its own log in ``<out>/logs/``; ``fold`` is the held-out fold
of a dataset with a ``cv`` block (Zheng68K) and ``None`` otherwise. A ``v2``
job writes both of v2's integrations (``v2`` and ``v2_vote``). The default
plan is one refinement round on the manuscript recipes, one run each:

1. The gate: LARRY with 2,000 HVGs, so a failing pipeline stops the queue
   before it spends hours on the rest.
2. iEP RNA with 2,000 HVGs, then Zheng68K's five held-out folds.

R1's runs live in their own folder (``--jobs`` with ``r1`` items).
A job starts only when the memory left once the running jobs reach their
expected peak covers its own (and is at least ``--min-free-gb``). A job's
expected peak is the largest of its dataset and method's finished runs, or
``PEAK_GB`` before one has finished. The next start waits ``--settle``
seconds so the new job's memory shows up first. If memory still runs out,
the kernel kills a job before other work on the machine. Touch
``<out>/STOP`` to stop launching jobs; running ones finish.
Rerunning the queue skips finished jobs (their record JSON exists) and
retries the rest. ``<out>/jobs.csv`` gets one row per finished attempt.
``--jobs`` replaces the plan above with an explicit list, run in order
without the gate. ``--registry`` points the jobs at another dataset registry
(the simulated datasets: ``sim_registry.py``); without ``--jobs`` the plan is
then every setup of that registry in its order, v2 and R1, seed 0 each.
``--default-peak-gb`` is the expected peak of a dataset with no finished run
and no ``PEAK_GB`` entry (each simulated dataset runs once per method, so
none of them ever has one).
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = [('larry_bench_hvg2000', 0, 'v2', None)]
ALL_METHODS = ('r1', 'v2')
#: Expected peak (GB) before a dataset's first finished run of a method: v2's ATAC runs peaked at
#: 19 GB; Zheng68K (55,000 training cells, 2,000 genes) is a guess.
PEAK_GB = {'iep_atac_peak': 20.0, 'zheng68k_hvg2000': 10.0}


def job_list() -> list:
    """One refinement round: v2 on the manuscript recipes, one run each."""
    return list(GATE) + [('iep_rna_hvg2000', 0, 'v2', None)] + [
        ('zheng68k_hvg2000', 0, 'v2', fold) for fold in range(5)]


def registry_jobs(path) -> list:
    """Every setup of a registry, in its order: v2, then R1, seed 0 each."""
    with open(path, encoding='utf-8') as fh:
        setups = [key for key in json.load(fh) if not key.startswith('_')]
    return [(setup, 0, method, None) for setup in setups for method in ('v2', 'r1')]


def parse_jobs(text) -> list:
    """``dataset:seed:method[:fold]`` items, comma-separated, as job tuples."""
    jobs = []
    for item in text.split(','):
        parts = item.strip().split(':')
        dataset, seed, method = parts[:3]
        assert method in ALL_METHODS, f'unknown method {method!r}'
        jobs.append((dataset, int(seed), method, int(parts[3]) if len(parts) > 3 else None))
    return jobs


def rep(job) -> str:
    _, seed, _, fold = job
    return f'seed{seed}' if fold is None else f'fold{fold}'


def is_done(out, job) -> bool:
    dataset, _, method, _ = job
    return os.path.exists(os.path.join(out, 'results', dataset, f'{method}_{rep(job)}.json'))


def free_gb() -> float:
    with open('/proc/meminfo') as fh:
        for line in fh:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2 ** 20
    raise RuntimeError('MemAvailable not found in /proc/meminfo')


def rss_gb(pid) -> float:
    """Resident memory of a process; 0 once it has exited."""
    try:
        with open(f'/proc/{pid}/status') as fh:
            for line in fh:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return 0.0


def expected_peak_gb(out, dataset, method, default=0.0) -> float:
    """The largest peak of the finished runs of this dataset and method, or ``PEAK_GB`` (else ``default``) before any."""
    peaks = []
    for path in glob.glob(os.path.join(out, 'results', dataset, f'{method}_*.json')):
        try:
            with open(path) as fh:
                peaks.append(json.load(fh)['peak_rss_gb'])
        except (OSError, ValueError, KeyError):
            continue
    return max(peaks) if peaks else PEAK_GB.get(dataset, default)


def prefer_as_oom_victim(pid):
    """If memory runs out, the kernel kills this job before other work on the machine."""
    try:
        with open(f'/proc/{pid}/oom_score_adj', 'w') as fh:
            fh.write('1000')
    except OSError:
        pass


class Queue:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.launch_lock = threading.Lock()
        self.stop = threading.Event()
        self.running = {}  # pid -> job
        os.makedirs(os.path.join(args.out, 'logs'), exist_ok=True)

    def log(self, message):
        with self.lock:
            print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {message}', flush=True)

    def stopped(self) -> bool:
        return self.stop.is_set() or os.path.exists(os.path.join(self.args.out, 'STOP'))

    def headroom_gb(self) -> float:
        """Memory left once every running job has grown to its expected peak."""
        with self.lock:
            running = list(self.running.items())
        growth = sum(max(0.0, self.peak_gb(dataset, method) - rss_gb(pid))
                     for pid, (dataset, _, method, _) in running)
        return free_gb() - growth

    def peak_gb(self, dataset, method) -> float:
        return expected_peak_gb(self.args.out, dataset, method, self.args.default_peak_gb)

    def need_gb(self, dataset, method) -> float:
        return max(self.peak_gb(dataset, method), self.args.min_free_gb)

    def run(self, job) -> bool:
        """Run one job; ``True`` if it finished. Returns ``False`` unstarted if the queue stops."""
        dataset, seed, method, fold = job
        name = f'{dataset}_{rep(job)}_{method}'
        with self.launch_lock:
            while self.headroom_gb() < self.need_gb(dataset, method):
                if self.stopped():
                    return False
                time.sleep(60)
            if self.stopped():
                return False
            self.log(f'start {name} ({free_gb():.1f} GB free, {self.headroom_gb():.1f} GB once running jobs peak)')
            started = time.time()
            with open(os.path.join(self.args.out, 'logs', f'{name}.log'), 'a') as log:
                proc = subprocess.Popen(
                    [sys.executable, '-I', os.path.join(HERE, 'run_job.py'), '--out', self.args.out,
                     '--dataset', dataset, '--seed', str(seed), '--method', method,
                     '--accelerator', self.args.accelerator, '--threads', str(self.args.threads)]
                    + ([] if fold is None else ['--fold', str(fold)])
                    + ([] if self.args.registry is None else ['--registry', self.args.registry])
                    + ([] if self.args.protocol is None else ['--protocol', self.args.protocol]),
                    stdout=log, stderr=subprocess.STDOUT, env=self.env(),
                )
            prefer_as_oom_victim(proc.pid)
            with self.lock:
                self.running[proc.pid] = job
            time.sleep(self.args.settle)
        rc = proc.wait()
        with self.lock:
            del self.running[proc.pid]
        minutes = (time.time() - started) / 60
        ok = rc == 0 and is_done(self.args.out, job)
        self.log(f'{"done" if ok else "FAILED"} {name} in {minutes:.1f} min (exit {rc})')
        with self.lock:
            path = os.path.join(self.args.out, 'jobs.csv')
            new = not os.path.exists(path)
            with open(path, 'a', newline='') as fh:
                writer = csv.writer(fh)
                if new:
                    writer.writerow(['dataset', 'seed', 'fold', 'method', 'status', 'exit', 'minutes', 'finished'])
                writer.writerow([dataset, seed, '' if fold is None else fold, method, 'ok' if ok else 'failed',
                                 rc, f'{minutes:.1f}', time.strftime('%Y-%m-%d %H:%M:%S')])
        return ok

    def env(self) -> dict:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HDF5_USE_FILE_LOCKING='FALSE')
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            env[name] = str(self.args.threads)
        return env

    def pool(self, jobs) -> list:
        """Run ``jobs`` with ``--workers`` workers; returns the jobs that failed."""
        pending = list(jobs)
        failed = []

        def worker():
            while not self.stopped():
                with self.lock:
                    if not pending:
                        return
                    job = pending.pop(0)
                if not self.run(job) and not self.stopped():
                    with self.lock:
                        failed.append(job)

        threads = [threading.Thread(target=worker) for _ in range(self.args.workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, help='Benchmark folder.')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--min-free-gb', type=float, default=6.0,
                        help='Start a job only when this much memory is available.')
    parser.add_argument('--settle', type=int, default=180,
                        help='Seconds between starting a job and checking memory for the next one.')
    parser.add_argument('--accelerator', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--jobs', default=None,
                        help='Comma-separated dataset:seed:method[:fold] items to run in order instead of the '
                             'default plan, without the gate.')
    parser.add_argument('--registry', default=None,
                        help="Dataset registry for the jobs (default: run_job.py's). Without --jobs, the plan "
                             'is every setup in it, v2 and R1, seed 0.')
    parser.add_argument('--protocol', default=None,
                        help='v2 protocol file for every job (an ablation arm), in place of revision/v2/protocol.json.')
    parser.add_argument('--default-peak-gb', type=float, default=0.0,
                        help='Expected peak (GB) of a dataset with no finished run and no PEAK_GB entry.')
    args = parser.parse_args(argv)

    queue = Queue(args)
    if args.jobs:
        plan = parse_jobs(args.jobs)
    else:
        plan = registry_jobs(args.registry) if args.registry else job_list()
    jobs = [job for job in plan if not is_done(args.out, job)]
    queue.log(f'{len(jobs)} of {len(plan)} jobs to run, {args.workers} workers, a job starts when the '
              f'memory left once running jobs peak covers its own peak (at least {args.min_free_gb} GB; '
              f'before a first finished run: {PEAK_GB})')
    gate = [] if args.jobs or args.registry else [job for job in jobs if job in GATE]
    if gate and queue.pool(gate):
        queue.log('a gate job failed; stopping (see logs/)')
        sys.exit(1)
    failed = queue.pool([job for job in jobs if job not in gate])
    queue.log(f'queue finished; {len(failed)} failed: {failed}' if failed else 'queue finished')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
