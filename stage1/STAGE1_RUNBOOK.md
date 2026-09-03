# SkyGym Stage-1 Hand-off Runbook

You are executing a fully-specified ML pipeline. No prior conversation context is
needed: every command, path, verification gate and expected output is below.
Do the steps **in order** and do not improvise on the "DO NOT CHANGE" contract (§8).

---

## 0. Mission brief (what & why)

SkyGym (repo `ARYAN2302/SkyGym`) is a drone-recon **synthetic data factory**: scripted
autopilots fly physics-honest trajectories; three simulated sensors (radar az/el/range,
EO camera bearing+stereo-range, RF az-only) emit *corrupted* detections; a Stone Soup
EKF tracker (the only tracker, `skygym/stone_soup.py`) consumes detections and produces
state estimates; hidden ground truth is used ONLY for scoring/labels.

**Stage-1 goal**: train the first learned models on top of this factory and beat the
classical baseline on held-out seeds:
- **Threat ID** — classify quad / fixed_wing / bird from the detection+track stream.
- **Path prediction** — predict future positions 1 s / 2 s / 4 s ahead.

Two rungs, in one script each:
| Rung | Model | Task |
|---|---|---|
| 1 | XGBoost on engineered 1 s-window features | threat ID |
| 2 | GRU (2×64) multi-task | threat ID + learned *corrections* to the tracker's CV extrapolation |

Data discipline: seed-blocked splits (train 1,000,000+ / val 8,000,000+ / test
9,000,000+ — zero overlap), features contain **no truth-derived values** (clutter flag
deliberately excluded), truth appears only as labels.

---

## 1. Setup (~5 min)

Requires: python ≥ 3.10 (validated on 3.12/3.13), git, internet for pip.

```bash
export PROJECT_ROOT=$HOME/skygym-run        # any absolute path works
mkdir -p $PROJECT_ROOT/scripts $PROJECT_ROOT/download/stage1/corpus $PROJECT_ROOT/download/stage1/models
cd $PROJECT_ROOT

# 1a. clone the repo — the four scripts ship INSIDE it under stage1/
git clone https://github.com/ARYAN2302/SkyGym $PROJECT_ROOT/skygym_repo

# 1b. copy the four bundled scripts out of the cloned repo
cp $PROJECT_ROOT/skygym_repo/stage1/*.py $PROJECT_ROOT/scripts/

# 1c. point the scripts at your project root — NO file edits needed.
#     Every script reads SKYGYM_PROJECT_ROOT (falls back to /home/z/my-project).
export SKYGYM_PROJECT_ROOT=$PROJECT_ROOT
#     NOTE: this must be set in every shell that runs the scripts.
#     Either prefix each command (SKYGYM_PROJECT_ROOT=$PROJECT_ROOT python3 ...) or:
#     echo "export SKYGYM_PROJECT_ROOT=$PROJECT_ROOT" >> ~/.bashrc
#     (Fallback: sed-rewriting the hardcoded default also works:
#      sed -i "s|/home/z/my-project|$PROJECT_ROOT|g" $PROJECT_ROOT/scripts/*.py)

# 1d. python environment (venv strongly recommended)
python3 -m venv $PROJECT_ROOT/.venv && source $PROJECT_ROOT/.venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib gymnasium stonesoup scikit-learn xgboost
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build; omit flag for CUDA GPU
```

Verified versions: stonesoup 1.9.1 and current-pip-latest both work; torch 2.14+cpu
works. The scripts are CPU-pure (no `.cuda()`); GPU optional and unnecessary.

## 2. Verify the vel_rmse metric fix (10 s — already applied upstream)

`skygym/stone_soup.py` used to compute `vel_rmse_mps` from the wrong row slice —
`r[4:7]` grabs `[est_u_m, est_vel_e, est_vel_n]` (altitude leaks into a velocity
error → nonsense ~113 m/s). This fix is now **pushed to the repo** (v0.3.2), so a
fresh clone already has it. Verify:

```bash
cd $PROJECT_ROOT/skygym_repo
grep -n "r\[5:8\]" skygym/stone_soup.py     # expect hits (fixed)
```

If your clone is old and still shows `r[4:7]`, apply the fix:

```bash
sed -i 's/np\.array(r\[4:7\])/np.array(r[5:8])/' skygym/stone_soup.py
```

(The corpus builder does NOT depend on this fix, but any future Stone Soup
summary runs do.)

Optional speed edit (more cores → faster corpus):

```bash
sed -i 's/with Pool(2) as pool:/with Pool(min(8, os.cpu_count() or 2)) as pool:/' \
    $PROJECT_ROOT/scripts/build_stage1_corpus.py
```

Sanity check of the whole stack (optional, ~40 s): `python3 scripts/bench_stage1.py`
→ expect 4 lines like `ep0 approach quad 6.5s tracked=100.0%`.

## 3. Build the corpus (~35–45 min on 2 cores; ~10 min on 8 cores)

```bash
cd $PROJECT_ROOT
python3 scripts/build_stage1_corpus.py 2>&1 | tee download/stage1/build.log
```

This generates 480 fully-seeded episodes (320 train / 80 val / 80 test; 20 s each,
class-balanced quad/fixed_wing/bird, mixed behaviours, noise 0.5–2.0×, clutter
0.5–2.5×), runs the Stone Soup fusion tracker on each, and writes per-frame records
to `download/stage1/corpus/{train,val,test}.jsonl` (~150 KB/episode, ~72 MB total).

**If your runner kills long foreground processes**, use the chunked equivalent
(each ≈ 5 min on 2 cores; run sequentially, `--append` is mandatory after the first):

```bash
python3 scripts/build_stage1_corpus.py --chunk train:0:80
python3 scripts/build_stage1_corpus.py --chunk train:80:160 --append
python3 scripts/build_stage1_corpus.py --chunk train:160:240 --append
python3 scripts/build_stage1_corpus.py --chunk train:240:320 --append
python3 scripts/build_stage1_corpus.py --chunk val:0:80    --append
python3 scripts/build_stage1_corpus.py --chunk test:0:80   --append
```

**Verify gate 1** (must pass before training):

```bash
wc -l download/stage1/corpus/*.jsonl          # expect: 320 / 80 / 80
python3 - <<'EOF'
import json, collections
for s in ("train", "val", "test"):
    eps = [json.loads(l) for l in open(f"download/stage1/corpus/{s}.jsonl")]
    print(s, len(eps), collections.Counter(e["true_class"] for e in eps))
EOF
```

Expect ~40/30/30 class split per split; seed bases 1M/8M/9M. Episode build time is
~3.5 s median / ~15 s max per episode on an idle 2-core box.

## 4. Train (~6–12 min CPU)

```bash
cd $PROJECT_ROOT
python3 scripts/train_stage1.py 2>&1 | tee download/stage1/train.log
```

Expected console pattern:
- `[rung1] building window features ...` (slowest part, 2–5 min of pure-python loops)
- `[rung1] fit ...` then `[rung1] val: {"acc": ..., "readout_baseline_acc": ...}`
- `[rung2] ep00..ep39` lines with `RMSE CV->GRU m: 1s A->B 2s C->D 4s E->F`
- `FINAL val: {...}`

Artifacts written to `download/stage1/models/`: `xgb_id.json`, `xgb_id_val.json`,
`gru_stage1.pt`, `gru_val.json`.

## 5. Evaluate on held-out test seeds (~2–4 min)

```bash
python3 scripts/eval_stage1.py 2>&1 | tee download/stage1/eval.log
```

Artifacts written to `download/stage1/`: **`scorecard.json`**, **`rmse_bars.png`**
(path-prediction RMSE: CV extrapolation vs GRU-corrected at 1/2/4 s),
**`id_confusion.png`** (3 confusion matrices: readout baseline / XGBoost / GRU).

## 6. Report back (mandatory)

Send the orchestrator exactly:
1. `wc -l download/stage1/corpus/*.jsonl`
2. `tail -25 download/stage1/train.log`
3. full `download/stage1/scorecard.json`
4. the two PNGs
5. any traceback if a step failed (with the step number)

**Success criteria**: GRU-corrected RMSE < CV-extrapolation RMSE at 2 s and 4 s
(a good run cuts ~15–25% at 2 s and ~25–50% at 4 s), and XGBoost/GRU threat-ID
accuracy clearly above the posterior-readout baseline (≥ +5 pts).

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `externally-managed-environment` pip error | use the venv from §1 (or add `--break-system-packages`) |
| `No module named stonesoup` | `pip install "stonesoup==1.9.1"` |
| torch download stalls | keep the `--index-url .../whl/cpu` flag |
| episode build times >> 20 s | machine overloaded; median is ~3.5 s/ep on idle 2 cores |
| corpus jsonl empty when training | build incomplete — recheck verify gate 1 |
| OOM during corpus build | 3 GB-RAM boxes: keep `Pool(2)`, close other jobs |

## 8. Contract — DO NOT CHANGE

- Seed blocks (`SPLITS` in build_stage1_corpus.py): train 1,000,000+ / val 8,000,000+ /
  test 9,000,000+. Changing them breaks the no-leakage guarantee.
- `HORIZONS = [10, 20, 40]`, `WARM = 5`, `WINDOW = 10` in train_stage1.py.
- `FEATURE_NAMES` ordering in build_stage1_corpus.py — it is the feature contract.
- Track-selection rule (most-mature live track) and masks (`track_valid`,
  `off > 500 m`, warm-up) must stay identical at train and test time.
- Truth is used ONLY as labels/targets — never as model input.

If the GRU does **not** beat CV extrapolation, allowed escalations in this order:
(1) GRU hidden 64→128, (2) epochs 40→80, (3) extend train split +320 episodes
(`--chunk train:320:320 --append` — seed base stays 1,000,000). Report which
escalations were used.
