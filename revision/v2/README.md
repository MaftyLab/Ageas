# Ageas v2: benchmark and ablation scripts

The label-free, feature-routed pipeline of Ageas v2 is part of the `ageas` package (`ageas.run_pipeline`; the method is described in `docs/v2.rst`). This folder holds what the revision ran it with:

- its frozen protocol;
- the benchmark scripts for the lineage-traced and simulated datasets;
- the R2.4 ablation.

## Protocol

`protocol.json` holds the settings every v2 benchmark ran under:

- `selection`: folds;
- `routing`: the pruning ladder;
- `synergy`: the no-harm mixture;
- `committee`: competence, committee and vote;
- `pipeline`: validation, second round, profile rules;
- `graph`: the graph stage.

Its `_history` records each change and the evidence behind it.

`run_pipeline`'s own defaults are simpler than the protocol (for example committee rule `mean`, no mixture in the vote). Pass the protocol's sections to run the method as benchmarked, as `run_job.py` (`run_v2`) does.

## Scripts

| Script | Does |
|---|---|
| `launch.sh` | Freezes the code on first launch, then starts or resumes the queue in the background (WSL) |
| `snapshot.py` | Copies this Ageas checkout into `<bench>/code` and checks the data against Ageas.share. Records the checkout's git state in `SNAPSHOT.json`: commit, uncommitted diff, and a hash of every untracked file |
| `run_queue.py` | Runs the jobs a few at a time. A job starts only when the memory left, once the running jobs reach their expected peak, covers its own. If memory still runs out, the kernel kills a job before other work. LARRY HVG seed 0 runs first as a gate. `--protocol` gives every job an ablation arm's protocol |
| `run_job.py` | Runs one (dataset, seed, method) job: v2 under `protocol.json`, or R1 under `revision/protocol.json`. Writes probabilities, members, out-of-fold record and run record (profile, rules, panel, strategy check, graph stage). `v2_final_<rep>.csv` is v2's output |
| `graph_stage.py` | Applies the graph stage to finished v2 runs and writes `v2_final_<rep>.csv`. It reads only their vote and every cell's features |
| `sim_registry.py` | Registry of the simulated datasets (`Ageas.share/data/simulation`: 90 RNA, 40 ATAC) and their query fates, for `run_job.py --registry` |
| `score_sim.py` | Scores v2 against R1 on the simulated datasets: per scenario and overall, paired by dataset |
| `score_baselines.py` | Scores v2's final output like the baseline methods in `Ageas.share/test.result` (same code, same cells): every simulated run (R1 too), and the real datasets with a bootstrap over cells |
| `ablate.py` | The R2.4 ablation. Rebuilds v2 with one component changed from its saved runs, adds the re-run arms (`run_queue.py --protocol`) and scores every variant. `--predictions <dir>` also writes each variant's per-cell predictions |
| `ablation_r2_4.md` | The ablation's design and results |

The scripts import `ageas` and `revision.common` from the checkout they sit in. The WSL `ageas` env holds an older `ageas` in site-packages, so always run them as scripts from this folder, never from an installed copy.

## Running (WSL)

Run outputs go to `~/ageas_work`, never into Dropbox. The simulated datasets were never used to build or refine v2, apart from the graph stage, which was tuned on seeds 1–5 only. Each dataset runs once per method (seed 0):

```bash
B=~/ageas_work/v2/sim
python -I revision/v2/sim_registry.py --out $B
bash revision/v2/launch.sh $B --registry $B/registry.json --workers 6 --threads 2 --settle 15 --min-free-gb 4 --default-peak-gb 4
python -I -B revision/v2/score_sim.py --bench $B
```

```bash
bash revision/v2/launch.sh                       # start or resume the default bench (~/ageas_work/v2/bench)
tail -n 20 ~/ageas_work/v2/bench/queue.log       # progress
touch ~/ageas_work/v2/bench/STOP                 # launch no more jobs
```

## Results

- **R2.4 ablation:** `ablation_r2_4.md` (design and results). The shared tables and per-cell predictions are in `Ageas.share/test.result/ablation`.
- **Comparison with the baseline methods:** `score_baselines.py`, against `Ageas.share/test.result/simulation_benchmark` and `test.result/real_data_baselines`.
