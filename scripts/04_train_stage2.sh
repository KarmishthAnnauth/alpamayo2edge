#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python -m distill.train_stage2 --config configs/default.yaml
