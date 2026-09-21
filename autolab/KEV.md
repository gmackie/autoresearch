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
