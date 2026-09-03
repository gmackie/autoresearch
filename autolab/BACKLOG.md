# BACKLOG.md — nanochat-speedrun

Ranked, ready-to-run hypotheses. One item = one `autolab run -H`. Keep ≥5 items
queued; when you run one, move it to Done with its experiment id and fold any
new fact into PROGRAM.md "Known truths". Operator items are not candidate
mutations and need a human (re-baseline, fixtures, evaluator changes).

Champion: none — run `autolab baseline` first

Context every item below assumes: the worker is the RTX 4070-class card (12 GB),
not the H100 the upstream defaults were tuned on (program.md reference:
val_bpb 0.9979, 953 steps, 500 M tokens in 300 s). Expect the baseline to
report far fewer `num_steps` and `total_tokens_M`; most of the early headroom is
"the recipe is sized for a GPU 6–8× faster than this one". Replace the
estimates in `why:` with the baseline's printed `num_steps` / `total_tokens_M` /
`mfu_percent` as soon as it exists. Metrics the evaluator writes: `val_bpb`
(promotion, ↓, min_effect 0.003), `training_seconds`, `peak_vram_mb`,
`mfu_percent`, `total_tokens_M`, `num_steps`, `num_params_M`.

Cost of every item: one 5-minute training run per seed (≈7–9 min wall with
startup, torch.compile and the 21 M-token eval; hard kill at 900 s).

## Queued (ranked by expected gain ÷ cost)

- [ ] **H1 · Smaller optimizer batch: TOTAL_BATCH_SIZE 2**19 → 2**17**
  - metric: `val_bpb` — expect −0.01 to −0.03; `num_steps` should roughly 4×. Headroom: baseline `num_steps` (if < 300, see why).
  - surface: `train.py:438` `TOTAL_BATCH_SIZE = 2**19`
  - change: `TOTAL_BATCH_SIZE = 2**17` (keep power of 2; `train.py:496` asserts it is a multiple of `DEVICE_BATCH_SIZE * 2048`, so DEVICE_BATCH_SIZE must be ≤ 64).
  - cost: 1 GPU run (5 min budget, ~8 min wall).
  - why: 524 K tokens/step at 4070 throughput (~1/6–1/8 of H100) is on the order of 100–150 optimizer steps in 300 s. The Muon momentum ramp alone is 300 steps (`train.py:527-529`, `frac = min(step / 300, 1)`) and the first 10 steps are excluded from the clock (`train.py:578`), so a step-starved run never reaches its tuned optimizer state. README (line 79) says exactly this for small platforms: lower TOTAL_BATCH_SIZE a lot, keep powers of 2. Largest single lever available; run it first.
  - run: `autolab run -H "Step-starved on the 4070: baseline num_steps is far below the 300-step Muon momentum ramp. TOTAL_BATCH_SIZE 2**19 -> 2**17 to take ~4x more optimizer steps in the same 5 min; expect val_bpb -0.01 to -0.03."`

- [ ] **H2 · Smaller model for a compute-starved budget: DEPTH 8 → 6**
  - metric: `val_bpb` — expect −0.005 to −0.02; `num_params_M` ↓, `total_tokens_M` ↑ ~30%. Headroom: tokens-per-parameter at baseline (total_tokens_M / num_params_M; H100 gets ~10).
  - surface: `train.py:450` `DEPTH = 8` (model_dim = DEPTH × ASPECT_RATIO rounded to HEAD_DIM, `train.py:469-477`)
  - change: `DEPTH = 6` (→ model_dim 384 → rounded to 512, 4 heads).
  - cost: 1 GPU run.
  - why: with ~1/7 of the H100's tokens in 5 min, a 50 M-param model sees ~1–2 tokens per parameter — far below any tokens/param sweet spot; a shallower, narrower model trains more steps and more tokens on a fixed budget. README line 77 names DEPTH as "the primary single knob"; `build_model_config` scales width, heads and the LR (`setup_optimizer` 1/√dmodel, `train.py:248`) automatically, so this is a one-line change. Try 4 next if 6 wins.
  - run: `autolab run -H "Compute-starved 5-min budget on a 4070: fewer tokens per parameter than any sweet spot. DEPTH 8 -> 6 (model_dim 512) to trade capacity for ~30% more tokens; expect val_bpb -0.005 to -0.02."`

- [ ] **H3 · Seed-noise floor: plumb AUTOLAB_SEED into torch.manual_seed**
  - metric: `val_bpb` — expect |Δ| ≈ 0.001–0.003 vs champion; the |Δ| is the result (σ sample that decides whether `min_effect` 0.003 is real). Headroom: none sought.
  - surface: `train.py:458-459` `torch.manual_seed(42)` / `torch.cuda.manual_seed(42)`
  - change: `SEED = int(os.environ.get("AUTOLAB_SEED", 42))` and seed both calls with it (harness exports `AUTOLAB_SEED` and `SEED` per seed, `autolab/src/autolab/engine.py:104`). With `seeds: [0]` this run uses seed 0 against the champion's 42.
  - cost: 1 GPU run.
  - why: the harness's paired-seed decision (`stats.py`: mean improvement > min_effect AND majority of seeds improve) is inert while `train.py` ignores the seed — every "seed" is the same run, and THREATS lists "Seed shopping" as open. research.yaml admits `min_effect: 0.003` is "roughly the observed noise band"; nobody has measured it here. Only init noise is seeded (the dataloader is deterministic), so this is a lower bound on σ. Expected outcome: `rejected` — good.
  - run: `autolab run -H "Measurement: read AUTOLAB_SEED into torch.manual_seed (was hardcoded 42) and change nothing else. Paired delta vs champion is an init-noise sample to test whether min_effect 0.003 is inside the noise band; also unblocks multi-seed runs. Expect rejected."`

- [ ] **H4 · Full-context attention on a non-Hopper GPU: WINDOW_PATTERN "SSSL" → "L"**
  - metric: `mfu_percent` / `num_steps` ↑ first; `val_bpb` −0.002 to −0.006 from the extra steps plus full context in every layer. Falsified if steps do not rise. Headroom: baseline `mfu_percent`.
  - surface: `train.py:435` `WINDOW_PATTERN = "SSSL"`
  - change: `WINDOW_PATTERN = "L"` (`_compute_window_sizes`, `train.py:195-206`, already forces the last layer to L).
  - cost: 1 GPU run.
  - why: README line 78: SSSL's banded attention "may be very inefficient" off-H100; on sm_89 the FA3 kernel comes from `kernels-community/flash-attn3` (`train.py:21-24`), whose sliding-window path is unproven on this card. Cheap, one token, and also a simplification if neutral (program.md: simpler at equal bpb is a keep).
  - run: `autolab run -H "SSSL banded attention may be slow on the non-Hopper FA3 kernel (README). WINDOW_PATTERN SSSL -> L: expect mfu_percent and num_steps up, val_bpb -0.002 to -0.006, and a simplification if neutral."`

- [ ] **H5 · Higher Muon matrix LR for a short run: MATRIX_LR 0.04 → 0.06**
  - metric: `val_bpb` — expect −0.003 to −0.008; falsified (LR already at the edge) if loss spikes or `val_bpb` rises. Watch the `FAIL` fast-fail (`train.py:569-572`) → `error`.
  - surface: `train.py:441` `MATRIX_LR = 0.04`
  - change: `MATRIX_LR = 0.06`; leave EMBEDDING_LR / UNEMBEDDING_LR alone (one knob).
  - cost: 1 GPU run.
  - why: with a few hundred steps or fewer, the schedule spends half its budget in warmdown (`WARMDOWN_RATIO = 0.5`, `train.py:446`) and never accumulates the H100's 953 steps of progress; short runs typically want a hotter peak. Upstream's own example log shows an LR increase as the first keep. Pair the result with H1: if H1 quadruples steps, the optimum shifts back down, so run this after H1 settles.
  - run: `autolab run -H "Few optimizer steps in 5 min on this GPU favour a hotter peak: MATRIX_LR 0.04 -> 0.06 (Muon matrices only). Expect val_bpb -0.003 to -0.008; a loss spike / FAIL means we are already at the LR edge."`

- [ ] **H6 · Progress-based Muon momentum ramp instead of a 300-step one**
  - metric: `val_bpb` — expect −0.002 to −0.005; neutral is also informative (momentum warmup does not matter at this scale). Headroom: only relevant while baseline `num_steps` < ~600.
  - surface: `train.py:527-529` `get_muon_momentum(step)`: `frac = min(step / 300, 1)`
  - change: ramp on time-progress instead, e.g. `frac = min(progress / 0.25, 1)` (0.85 → 0.95 over the first quarter of the budget) and pass `progress` at `train.py:557`.
  - cost: 1 GPU run.
  - why: every other schedule in the file is already progress-based (`get_lr_multiplier`, `get_weight_decay`, `train.py:518-532`); momentum is the one step-based schedule, tuned for the H100's step count. If the 4070 baseline reports fewer than 300 steps, Muon spends the whole run below its tuned 0.95 momentum. Drop this item if H1 pushes `num_steps` well past 300.
  - run: `autolab run -H "get_muon_momentum ramps over 300 steps, tuned for H100 step counts; baseline on this GPU never reaches 0.95. Make the ramp progress-based (0.85->0.95 over the first 25% of the time budget) like every other schedule in train.py; expect val_bpb -0.002 to -0.005."`

- [ ] **H7 · Simplification: WEIGHT_DECAY 0.2 → 0.0 (delete the decay schedule if neutral)**
  - metric: `val_bpb` — expect −0.003 to +0.003 (neutral-to-slightly-better); a keep at equal bpb under program.md's simplicity criterion, but the harness needs > min_effect, so record the Δ in Known truths either way.
  - surface: `train.py:443` `WEIGHT_DECAY = 0.2` (+ `get_weight_decay`, `train.py:531-532`, cautious WD in `muon_step_fused`, `train.py:352-353`)
  - change: `WEIGHT_DECAY = 0.0`; nothing else.
  - cost: 1 GPU run.
  - why: the run trains < 1 epoch on the 10 downloaded shards (train loop prints `epoch`), tens of millions of tokens on a 50 M-param model — nowhere near the overfitting regime WD is for, and cautious decay costs an extra mask/multiply per Muon step. If neutral, this is a candidate for a later "delete get_weight_decay" simplification run.
  - run: `autolab run -H "Sub-epoch, compute-starved run: weight decay is a regulariser for a regime we are not in. WEIGHT_DECAY 0.2 -> 0.0; expect val_bpb within +-0.003 (neutral = simplification lead), better if decay was fighting the short schedule."`

## Operator (needs a human; not `autolab run`)

- [ ] **Baseline run (`autolab baseline`) — none exists.** Before it can go green on a 12 GB card: (a) `DEVICE_BATCH_SIZE = 128` (`train.py:451`) materialises fp32 logits of 128 × 2048 × 8192 × 4 B ≈ 8.6 GB (`train.py:283-284`) and will very likely OOM — the "baseline" for this hardware needs DEVICE_BATCH_SIZE lowered (32 or 16; comment says "reduce if OOM") and that fact recorded in the baseline hypothesis; (b) `get_kernel(...)` (`train.py:20-24`) fetches FA3 from the HF Hub, but `allow_network: false` and `read_allow` only covers `~/.cache/autoresearch` — pre-populate and allow the HF kernels cache (moot while the Linux box runs sandbox `none`, real once bwrap lands).
- [ ] **No-change variance run → set `min_effect`.** After H3 lands the seed plumbing, set `seeds: [0, 1, 2]`, re-baseline the champion on those seeds (paired comparison needs common seeds), and set `promotion.min_effect ≥ 3σ` (0.003 is a guess per research.yaml). Triples the cost of every experiment (~25 min).
- [ ] **Hidden promotion shard + `promotion.hidden_reveal`.** No `autolab/datasets/hidden` exists, so `autolab promote` has nothing to check and THREATS "Training on the validation shard" (the val parquet sits in the readable cache) is open. Rotate a held-out shard the candidate cannot read; store champion hidden scores.
- [ ] **Evaluator hardening (THREATS "Fabricated metrics", "Ignoring the time budget").** Have train.py save the final checkpoint and the frozen evaluator recompute `prepare.evaluate_bpb` on it, plus harness wall-clock accounting instead of self-reported `training_seconds`; bump `eval_version` (`nanochat-eval@0.1.0` → 0.2.0) and re-baseline.
- [ ] **Confirm compile caches on the worker.** `write_allow` lists `~/.cache/torchinductor`, `~/.triton`, `~/.nv`; if they are not actually writable each run recompiles two `fullgraph=True` optimizer kernels and the model (excluded from the 300 s budget but counted against the 900 s kill).

## Done / dropped

- nothing run yet
