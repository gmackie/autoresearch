# THREATS.md — nanochat-speedrun

Known ways a candidate could game `evaluate()`. Each discovered exploit
becomes a regression test.

## Open

- [ ] **Training on the validation shard.** The val parquet lives in the same
  readable cache (`~/.cache/autoresearch/data`); a candidate could add it to
  the training set and memorize it. Mitigation (planned): hidden promotion
  shard, rotated, excluded from the candidate-readable cache.
- [ ] **Ignoring the time budget.** `training_seconds` is self-reported.
  Current mitigations: `reported_training_budget` gate (10% slack) and the
  harness-level 900 s wall-clock kill. A candidate that trains 800 s and
  reports 300 s beats the gate — closed only by trusted wall-clock accounting, or by moving the budget loop out of train.py.
- [ ] **Seed shopping.** Initialization now reads `AUTOLAB_SEED` and the harness
  pairs three seeds. Candidate code can still ignore it or train several
  initializations. A manifest seed match is not proof of execution compliance.
- [ ] **Resource exhaustion in checkpoint decoding.** `weights_only=True`,
  bounded files, bounded config, strict state loading and the harness timeout
  limit accidental failures. They are not a general sandbox for malicious
  tensor archives or vulnerabilities in PyTorch/native kernels.

## Closed

- [x] **Fabricated BPB as the authoritative score.** The evaluator recomputes
  BPB from data-only tensor checkpoints using frozen inference and pinned
  `prepare.py`. A false report cannot improve the measured metric and a mismatch
  fails the run. Tests cover fabricated numbers, invalid metadata and symlinks.
  This is limited to explicitly supported architectures; it does not establish
  clean training data or honest training time.

- [x] **Editing the evaluator or launcher.** `autolab/**` is outside the
  mutable globs; the harness marks any such diff `invalid` before running
  (covered by harness test `test_immutable_edit_is_invalid`).
- [x] **Reading hidden datasets at run time.** Sandbox denies reads/writes
  on `autolab/datasets/hidden` (harness sandbox proof tests).
