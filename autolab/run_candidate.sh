#!/bin/sh
# Frozen launcher (autolab/** is immutable). Runs the candidate's train.py
# with the venv shared from the control checkout.
set -e
if [ ! -x .venv/bin/python ]; then
    echo "no .venv in worktree — run 'uv sync' in the control checkout first" >&2
    exit 2
fi
exec .venv/bin/python train.py
