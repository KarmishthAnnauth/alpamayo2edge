#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python -m distill.train_stage1 --config configs/default.yaml
