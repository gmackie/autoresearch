# Kev-assisted research

Install the Autolab version containing `autolab ideas`. Keep Kev in its own
Python environment: its Torch constraint conflicts with this trainer's 2.9.1.
Use this workflow from a writable research control worktree.

```sh
mkdir -p autolab/results/proposals
autolab ideas brief > autolab/results/proposals/brief.json
```

Use the six prompts to generate distinct optimizer, architecture, throughput,
simplification, near-miss and unconventional hypotheses. Read train.py and
BACKLOG.md to ground them in real evidence. Return a batch with the brief's
context and proposal schema, recording the generator/model configuration.
Save the batch under `autolab/results/proposals/batch.json`.

```sh
autolab ideas rank autolab/results/proposals/batch.json \
  --endpoint http://127.0.0.1:8009 --checkpoint 'checkpoint-sha256:REPLACE' --seed 42
```

Rankings are advisory. Choose from four ranked ideas plus two exploration slots;
measurement ideas remain a separate lane. Keep the existing numerical evaluator
and promotion policy in charge. Implement one hypothesis in train.py, run it,
and link the observed result:

```sh
autolab run -H 'H1: hypothesis, parent evidence and expected diagnostic movement'
autolab ideas link REPORT_ID H1 EXPERIMENT_ID
autolab ideas export > autolab/results/proposals/outcomes.jsonl
```

The link attests implementation fidelity. Recreate the brief and re-rank after
a champion/contract change. Never modify source or install Kev into this trainer
just to obtain scores. Reports under results are excluded from snapshots.

These labels remain provisional until THREATS.md's seed, evaluation-integrity,
and hidden-validation issues are resolved. Do not interpret `rejected` as a bad
idea: the export uses comparable paired measurements and omits diagnostic runs.

For improving Kev itself, use Autolab's separate `examples/kev-selector`
environment. Curate train/development/promotion splits by campaign and family,
fine-tune from a released checkpoint, and freeze the selected version for each
research campaign. An outcome-trained model can be selected with `rank
--strategy outcome`; pretrained models default to four rubric questions.

## First live ranking

On 2026-09-21, pinned `jaredpalmer/kev-0.8b` revision
`c917edefdfd72b3e9ba71455584700acc70595f6` scored 24 authored hypotheses
on MPS. The [complete report](../docs/validation/2026-09-21-kev-ranking.json)
contains the pool, API requests/responses and advisory scores. The shortlist was:

- H09: lower DEVICE_BATCH_SIZE to fit the worker (ranked).
- H10: compare device batches 16 and 32 after establishing a fitting baseline (ranked).
- H24: halve embedding LR with matrix LR fixed (ranked).
- H06: test depth four (ranked).
- H02: higher matrix LR (exploration).
- H13: zero weight decay (exploration).

H12, H19 and H20 remained measurement work. These are proposed experiments,
not completed training runs. Respect prerequisites: H10 depends on first finding
a fitting baseline. Some ideas are conditional followups, not observed near misses.
The archived context predates this documentation commit; create a fresh brief
before further ranking. This result does not reorder the authoritative backlog.

A dedicated Autolab service is now installed at `http://127.0.0.1:8010` under
LaunchAgent `io.autolab.kev`. Pass `--endpoint http://127.0.0.1:8010` to rank
against it. The archived report above was reproduced against this endpoint.
The service retains the released model: a separate real fine-tuning pilot was
rejected for no ranking gain and slightly worse calibration.

## Run an autonomous campaign

The new `autolab campaign` command automates six idea-generation passes, Kev
ranking, isolated implementation, experiments, and development-result feedback.
It uses your installed Codex CLI and configured model by default. On a CUDA
worker with this environment's dependencies and data, establish a fresh baseline
and run a bounded campaign:

```sh
autolab baseline
autolab campaign \
  --endpoint http://127.0.0.1:8010 \
  --checkpoint 'jaredpalmer/kev-0.8b@c917edefdfd72b3e9ba71455584700acc70595f6' \
  --max-experiments 6 --budget-seconds 3600 --stage-timeout 180
```

The endpoint must be reachable from that worker; the localhost service above is
on the development Mac. Use a forwarded or otherwise reachable endpoint on
remote workers. Budgets count failed attempts too. Your checkout stays unchanged;
experiments start from the champion and only declared mutable proposal paths can
be patched. Use `autolab branch <experiment-id>` to inspect a winner.

Inspect `autolab/results/campaigns/<id>/state.json`, `learning.jsonl`, and
`outcomes.jsonl`. Each round receives recent development measurements; outcome
labels retain automated, unreviewed patch provenance. Curate these before training
Kev. The campaign does not automatically deploy new classifier weights. A live
CPU toy acceptance validates orchestration; no nanochat campaign efficiency gain
has yet been measured. The evaluator issues listed above still apply.

## RTX 4070 Ti worker profile

`autolab/profiles/rtx4070-checkpoint.patch` is a reproducible operator setup for the
12 GB worker. Apply it in a **fresh writable checkout**, commit, and prepare data:

```sh
git apply autolab/profiles/rtx4070-checkpoint.patch
git add train.py autolab/research.yaml
git commit -m 'Configure RTX 4070 checkpoint baseline'
uv sync
uv run python prepare.py --num-shards 2 --download-workers 2
autolab baseline
```

The profile uses depth 4, device batch 8, 65,536 tokens per optimizer step,
`AUTOLAB_SEED`, and portable worktree placement. It preserves the training clock
and independently recomputes development BPB from checkpoints over three paired
seeds. Treat this as its own baseline, not a comparison with the upstream H100
configuration or the earlier printed-metric pilot. See [PROGRAM.md](PROGRAM.md)
for supported architectures and [THREATS.md](THREATS.md) for remaining limitations.

On the worker, Kev runs on demand as `autolab-kev-gpu.service`. Autolab's
`--rank-start-command` / `--rank-stop-command` hooks start it only for ranking,
then release its GPU allocations before experiments. The `ssh_agent_relay.py` and
`spool_adapter.py` helpers let the Mac run Codex without copying credentials onto
the GPU worker. See Autolab's `docs/gpu-campaigns.md` for complete commands.

The [first GPU campaign record](../docs/validation/2026-09-21-gpu-campaign.json)
contains a 1.162591 BPB baseline and a Kev-selected value-embedding gate removal
that scored 1.180244 and was rejected. Both used 300.1 training seconds, but achieved
different token counts (229.4M versus 208.9M). One provisional negative label was
exported. Kev was inactive in all 75 sampled training states. The initial full-file
implementation timeout and exact-edit retry are documented in the evidence.

[Checkpoint validation evidence](../docs/validation/2026-09-21-checkpoint-evaluation.json) records exact CUDA inference parity,
full-shard recomputation at **1.226453216 BPB**, and rejection of a fabricated
**0.01** report with the measured BPB unchanged. The candidate originally reported
1.226452083 (difference 0.00000113). Pure contract tests: 19 passed; CUDA smoke:
1 passed; Autolab tests: 103 passed; package build passed. These checks validate
the scorer, not research efficacy. No new classifier weights or learning labels
were produced from the contended run.
