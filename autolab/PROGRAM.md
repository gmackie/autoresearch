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
- **Authoritative measurement:** the frozen evaluator loads a data-only checkpoint
  and recomputes `val_bpb` on CUDA at batch size 8. The candidate's reported BPB
  is only a consistency check (absolute tolerance 0.0005).
- **Mutable (yours):** `train.py`, and only `train.py`. Optimizer, schedules,
  batch sizes and training changes are supported. Architecture changes must
  conform to the frozen checkpoint contract below. Arbitrary candidate forward
  code is never executed by the evaluator.
- **Immutable (not yours):** `prepare.py` (`TIME_BUDGET` 300 s, `MAX_SEQ_LEN`,
  tokenizer, dataloader, `evaluate_bpb`), `autolab/**` (launcher, evaluator,
  research.yaml), `pyproject.toml` (no new dependencies — you have what is in
  the shared `.venv`). Any diff outside `train.py` makes the experiment
  `invalid` before it runs.
- **Hard gates:** `checkpoint_evaluated`, `reported_bpb_matches`, and
  `reported_training_budget` (self-reported seconds > 0 and ≤ 330), plus harness
  exit/time gates. Missing checkpoints, invalid configs, incompatible weights,
  and nonfinite results fail closed.
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
and every champion is provisional. Checkpoint recomputation is implemented; a
rotated hidden shard that candidates cannot read remains future work (THREATS.md).
A checkpoint can still benefit from validation contamination, so its independently
measured development score is not an unbiased held-out result. Treat the val parquet in `~/.cache/autoresearch/data` as
off-limits for training even though the sandbox does not yet enforce it —
overfitting it is a THREATS item, not a technique.

## Known truths (do not re-derive)

Facts from the code as it stands; append experiment-derived truths with
their ids as runs complete.

- The 300 s clock counts only steps after the first 10 (`train.py:578`,
  `603`); compile and warm-up are free for the budget but count toward the
  900 s harness kill.
- Initialization reads `AUTOLAB_SEED` (default 42 outside the harness).
  The contract uses paired seeds `[0, 1, 2]`; the dataloader is deterministic.
  This provides repeated measurements, not proof that candidate code honored the seed.
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
- Candidate summary output remains useful diagnostics; it is not parsed into the
  authoritative score. The evaluator consumes `checkpoint.pt` and `checkpoint.json`.
- Read `autolab/THREATS.md`: training time, seed compliance, validation visibility,
  and desktop GPU contention remain limitations.

## Checkpoint contract

Save a plain mapping of parameter names to detached CPU tensors in
`$AUTOLAB_ARTIFACT/checkpoint.pt`, using `torch.save`. No modules or custom objects.
`checkpoint.json` must declare `schema_version: 1`, integer `seed` equal to
`AUTOLAB_SEED`, numeric `reported_bpb`, `reported_training_seconds`, and `config`.
The configuration has exactly these fields:

- `sequence_len`: 2048; `vocab_size`: 8192.
- `n_layer`: 1–12; `n_embd`: 128–1024; `n_head` and `n_kv_head`: 1–16.
  Width must divide evenly into heads; query heads must divide evenly into KV
  groups. Head dimension must be a multiple of 8 and at most 256.
- `window_pattern`: 1–12 uppercase S/L characters.
- `value_embedding_gates`: boolean. False means fixed `v += ve`; remove gate
  tensors from the model and export this flag as false for that variant.

The architecture must match `autolab/evaluator/frozen_model.py`, including MLP,
normalization, rotary embeddings and logit softcap. Strict tensor loading rejects
missing/unexpected/shape-mismatched weights. Unsupported architecture proposals
require an operator-reviewed evaluator revision and a fresh baseline, not candidate
code execution. The operator pins `prepare.py` via `prepare.sha256` inside the
fingerprinted evaluator directory; an unpinned scoring change fails evaluation.
Both checkpoint files must be regular files, not symlinks; limits are 2 GB for
weights and 16 KiB for metadata. PyTorch uses `weights_only=True`.

For the RTX 4070, apply `autolab/profiles/rtx4070-checkpoint.patch` to a fresh
writable checkout and commit before baselining. The old `rtx4070-pilot.patch`
is historical, intended for commit `fe9ee26`, and uses the old printed-metric
contract; do not apply it to the checkpoint evaluator.

## Validate the evaluator

Run `python -m pytest tests/test_checkpoint_contract.py -q` in an environment
with pytest; these tests do not require Torch. On the CUDA worker, run
`.venv/bin/python tests/checkpoint_cuda_smoke.py` for exact training/frozen-model
logit parity, mixed-dtype checkpoint loading and both gate variants.
Full-shard acceptance must also compare reported/recomputed BPB and reject a
copied checkpoint manifest with a fabricated score; retain this separately from
research outcomes. Never label a contended or incomplete comparison as a win/loss.

## Backlog

Ranked, ready-to-run hypotheses live in `autolab/BACKLOG.md` (one item per
`autolab run -H`, each with metric, surface, change, cost and evidence; an
Operator section for work that needs a human; Done keyed by experiment id).
Take the top item, run it, move it to Done, and fold what you learned into
"Known truths" above. Keep at least five items queued.

## Kev-assisted proposal selection

An optional shotgun brainstorming and ranking workflow is documented in
[KEV.md](KEV.md). Use `autolab ideas` to generate/score a pool upstream of this
loop, retain exploration and measurement slots, and link executed proposals to
actual results. Scores are advisory; the frozen evaluator still decides
promotion. Re-rank whenever the champion changes.
