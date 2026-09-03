# THREATS.md — nanochat-speedrun

Known ways a candidate could game `evaluate()`. Each discovered exploit
becomes a regression test.

## Open

- [ ] **Fabricated metrics block.** train.py prints its own `val_bpb:` line;
  a candidate can print any number. Mitigation (planned): train.py saves the
  final checkpoint; the evaluator re-runs `prepare.evaluate_bpb` (frozen
  code) on it and rejects if self-reported and recomputed val_bpb diverge.
- [ ] **Training on the validation shard.** The val parquet lives in the same
  readable cache (`~/.cache/autoresearch/data`); a candidate could add it to
  the training set and memorize it. Mitigation (planned): hidden promotion
  shard, rotated, excluded from the candidate-readable cache.
- [ ] **Ignoring the time budget.** `training_seconds` is self-reported.
  Current mitigations: `time_budget_respected` gate (10% slack) and the
  harness-level 900 s wall-clock kill. A candidate that trains 800 s and
  reports 300 s beats the gate — closed only by checkpoint re-evaluation
  plus wall-clock accounting, or by moving the budget loop out of train.py.
- [ ] **Seed shopping.** Single seed today; upstream train.py is not
  seed-parameterized. Add SEED plumbing, then paired multi-seed runs.

## Closed

- [x] **Editing the evaluator or launcher.** `autolab/**` is outside the
  mutable globs; the harness marks any such diff `invalid` before running
  (covered by harness test `test_immutable_edit_is_invalid`).
- [x] **Reading hidden datasets at run time.** Sandbox denies reads/writes
  on `autolab/datasets/hidden` (harness sandbox proof tests).
