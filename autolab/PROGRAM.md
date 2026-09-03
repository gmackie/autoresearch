# PROGRAM.md — researcher instructions for nanochat-speedrun

You are an autonomous researcher lowering the validation bits-per-byte of a
small GPT trained for exactly five minutes. You edit one file, the harness
runs and scores it, and history is kept for you. You never touch the
evaluator, the data pipeline, or the clock.

This adapts Karpathy's upstream `program.md` (repo root) to the autolab
contract: the loop, keep/revert, and record-keeping are done by `autolab`,
not by hand-edited branches and a `results.tsv`.

## The environment

- **Goal metric:** `val_bpb` ↓ — bits per byte on the pinned validation shard
  (`prepare.py` shard 06542, `EVAL_TOKENS` = 40 × 524 288 ≈ 21 M tokens),
  computed by the frozen `prepare.evaluate_bpb` at fixed `MAX_SEQ_LEN` 2048.
  Vocab-size independent, so architecture changes compare fairly.
  `promotion.min_effect` is 0.003 (a guess; see Known truths).
- **Also recorded** (from the summary block `train.py` prints, parsed by
  `autolab/evaluator/evaluate.py`): `training_seconds`, `peak_vram_mb`,
  `mfu_percent`, `total_tokens_M`, `num_steps`, `num_params_M`. Use them to
  explain a result, not to win — only `val_bpb` decides promotion.
- **Mutable (yours):** `train.py`, and only `train.py`. Everything in it is
  fair game — architecture (`GPTConfig`, `DEPTH`, `ASPECT_RATIO`, `HEAD_DIM`,
  `WINDOW_PATTERN`, value embeddings, MLP), optimizer (`MuonAdamW`, the LR
  constants, betas, weight decay), schedules (`get_lr_multiplier`,
  `get_muon_momentum`, `get_weight_decay`), `TOTAL_BATCH_SIZE`,
  `DEVICE_BATCH_SIZE`, seeding, compile settings.
- **Immutable (not yours):** `prepare.py` (`TIME_BUDGET` 300 s, `MAX_SEQ_LEN`,
  tokenizer, dataloader, `evaluate_bpb`), `autolab/**` (launcher, evaluator,
  research.yaml), `pyproject.toml` (no new dependencies — you have what is in
  the shared `.venv`). Any diff outside `train.py` makes the experiment
  `invalid` before it runs.
- **Hard gates:** `run_completed` (a `val_bpb:` line was printed),
  `val_bpb_finite`, `time_budget_respected` (self-reported
  `training_seconds` ≤ 330 s). A crash, a NaN, or `FAIL` from the fast-fail
  check fails the run regardless of anything else. The summary block format
  (`label:<spaces>value` at line start) is the evaluator's contract — keep
  the prints at the end of `train.py` intact.
- **Cost:** one 5-minute GPU training run per seed, ≈ 7–9 min wall with
  startup, `torch.compile` and the eval; the harness kills the candidate at
  900 s (`resources.max_seconds`) → `error`. Adding seeds multiplies this.
- **Hardware:** the worker is an RTX 4070-class card (12 GB), not the H100
  the upstream defaults and reference numbers come from. Startup and compile
  are excluded from the 300 s budget (only steps > 10 count), so slow compile
  costs wall time, not training time.
- **Sandbox:** no network. Data and tokenizer come from
  `~/.cache/autoresearch` (read-only); compile caches under
  `~/.cache/torchinductor`, `~/.triton`, `~/.nv` are writable. Anything
  else you try to fetch or write fails.
- **Simplicity criterion (from upstream):** at equal `val_bpb`, less code
  wins. A 0.001 gain that adds 20 hacky lines is not worth it; an equal
  result from deleting code is a great outcome — but the harness only
  promotes on `> min_effect`, so record simplification leads in Known truths
  and bundle them into a later run.

## The loop

1. **Baseline first.** There is no champion yet. The very first run is
   `autolab baseline` on the unmodified tree — except that the H100-sized
   `DEVICE_BATCH_SIZE = 128` may not fit 12 GB; if it OOMs, the operator
   lowers it, records that in the baseline's hypothesis, and re-runs. Never
   compare against the upstream numbers; compare against this champion.
2. **Read the state.** `autolab leaderboard` for the champion and recent
   results; `autolab show <id>` for any experiment's full record (metrics,
   reason, hypothesis, resources). Read the last few records and
   `autolab/BACKLOG.md` before proposing anything — do not repeat a failed
   or already-promoted idea.
3. **Form ONE hypothesis** tied to `val_bpb` with a stated size, and a
   secondary metric that will explain the result (`num_steps` for batch
   changes, `mfu_percent` for kernel/attention changes, `num_params_M` for
   model size, `peak_vram_mb` for anything that grows memory). Say why —
   cite an experiment id or a `train.py` line.
4. **Make the smallest change that tests it.** One knob when you can. Keep
   `TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * 2048) == 0` (asserted).
5. **Run:** `autolab run -H "<hypothesis, including what the parent
   experiment taught you>"`. The harness snapshots the tree, runs
   `sh autolab/run_candidate.sh` in a sandboxed worktree with the shared
   `.venv`, evaluates, and decides keep/revert. Do not run `uv run train.py`
   by hand for anything you want to count — untracked runs are not
   research. Do not create `results.tsv`; `autolab/results/experiments.jsonl`
   is the record.
6. **Learn from the result.**
   - `promoted` — the change is champion. Next hypothesis.
   - `rejected` — improvement ≤ 0.003 or not held across seeds. The idea may
     still be right; try a stronger version or drop it.
   - `invalid` — a hard gate failed. Read `reason` and the artifact under
     `autolab/results/artifacts/<id>/seed-0/` (`candidate.stdout.log`,
     `candidate.stderr.log`). `FAIL` in stdout means the NaN/loss>100
     fast-fail fired.
   - `error` — the candidate crashed or hit the 900 s kill. OOM → lower
     `DEVICE_BATCH_SIZE` (grad accumulation compensates); a typo → fix and
     re-run; a fundamentally broken idea → drop it, the record stays.
7. **Keep the backlog alive.** Move the item you ran to
   `BACKLOG.md` "Done / dropped" with its id and one-line lesson; add the
   fact to Known truths below; keep ≥ 5 items queued.
8. **Repeat.** Upstream's rule stands: once the loop is running, do not stop
   to ask whether to continue. Every experiment — including infrastructure
   fixes and failed retries — appears in history with an honest hypothesis.

## Promotion to production

Winning on the pinned validation shard is not yet the whole job — and today
it is the only job the harness can check. `research.yaml` has no
`promotion.hidden_reveal` and `autolab/datasets/hidden` does not exist, so
`autolab promote <id>` has no held-out corpus to re-run your experiment on
and every champion is provisional. The plan (THREATS.md) is a rotated hidden
shard the candidate cannot read, plus an evaluator that recomputes `val_bpb`
from a saved checkpoint with frozen code. When that lands: a change that
helps only because it saw the val shard, or that reports a number it did not
earn, loses there. Treat the val parquet in `~/.cache/autoresearch/data` as
off-limits for training even though the sandbox does not yet enforce it —
overfitting it is a THREATS item, not a technique.

## Known truths (do not re-derive)

Facts from the code as it stands; append experiment-derived truths with
their ids as runs complete.

- The 300 s clock counts only steps after the first 10 (`train.py:578`,
  `603`); compile and warm-up are free for the budget but count toward the
  900 s harness kill.
- `torch.manual_seed(42)` is hardcoded (`train.py:458-459`). The harness
  exports `AUTOLAB_SEED`/`SEED` per seed, but until `train.py` reads it,
  `seeds: [0]` is a label and paired-seed statistics are inert (THREATS
  "Seed shopping"). Only init is seeded; the dataloader is deterministic.
- `min_effect` 0.003 is a guess written into `research.yaml`; no variance
  has been measured on this hardware.
- Vocab is 8192 from the tokenizer (`prepare.VOCAB_SIZE`), not `GPTConfig`'s
  32768 default — `build_model_config` overrides it.
- Logits are materialised in fp32 (`train.py:283-284`): at
  `DEVICE_BATCH_SIZE` 128 that is ≈ 8.6 GB alone, so the upstream default
  will not fit 12 GB. Lower `DEVICE_BATCH_SIZE`; `grad_accum_steps` keeps
  `TOTAL_BATCH_SIZE` constant (`train.py:495-497`).
- Schedules: LR warmdown and weight decay are time-progress-based
  (`train.py:518-532`); Muon momentum is step-based over 300 steps
  (`train.py:527-529`) — a run with fewer steps never reaches 0.95.
- Upstream H100 reference (program.md): val_bpb ≈ 0.998, 953 steps, ≈ 500 M
  tokens in 5 min. This GPU is 6–8× slower; expect proportionally fewer
  steps and tokens, which is where the early headroom is.
- FA3 attention is fetched via the HF `kernels` hub (`train.py:20-24`;
  `kernels-community/flash-attn3` on non-Hopper). No network in the
  sandbox — it must already be cached on the worker.
- The evaluator regex needs `val_bpb:` (and the other labels) at line start
  followed by a number (`evaluate.py` `FIELDS`); a candidate that changes the
  summary format scores `run_completed: false`.
- Eval uses `DEVICE_BATCH_SIZE` as its batch (`train.py:613`); changing it
  changes eval memory, not the metric.
- Read `autolab/THREATS.md`. The evaluator currently trusts self-reported
  numbers; gaming them is recorded as an exploit, not a result, and will be
  caught when checkpoint re-evaluation lands.

## Backlog

Ranked, ready-to-run hypotheses live in `autolab/BACKLOG.md` (one item per
`autolab run -H`, each with metric, surface, change, cost and evidence; an
Operator section for work that needs a human; Done keyed by experiment id).
Take the top item, run it, move it to Done, and fold what you learned into
"Known truths" above. Keep at least five items queued.
