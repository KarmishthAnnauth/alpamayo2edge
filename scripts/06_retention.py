"""Retention pair (D-025): the other half of the headline result.

Two invocations, in this order:

  # ONCE, on the untrained Cosmos3-Edge checkpoint, BEFORE stage 1:
  python scripts/06_retention.py baseline

  # after any stage-1 / stage-2 checkpoint:
  python scripts/06_retention.py score --ckpt /data/runs/stage2/epoch7

The baseline caches the base model's own greedy answers plus its top-k
next-token log-probs, so scoring a trained checkpoint never needs both models
resident at once (48GB card). Running `baseline` after training would measure
the trained model against itself and always report zero drift; the script
refuses to overwrite an existing baseline for exactly that reason.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, "src")
import torch
from distill.config import load_config
from distill.student.edge_wrapper import EdgeStudent
from distill.eval import retention

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["baseline", "score"])
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--ckpt", help="trained checkpoint dir (score mode)")
ap.add_argument("--force", action="store_true", help="overwrite an existing baseline")
a = ap.parse_args()

cfg = load_config(a.config)
rcfg = cfg.eval.retention
out = Path(rcfg.baseline_dir)

if a.mode == "baseline":
    if (out / retention.BASELINE_NAME).exists() and not a.force:
        sys.exit(f"{out / retention.BASELINE_NAME} already exists. A baseline must "
                 "come from the UNTRAINED checkpoint — pass --force only if you are "
                 "certain this student is still the base model.")
    student = EdgeStudent(cfg).cuda().eval()
    retention.save_baseline(retention.build_baseline(student, cfg), out)
    sys.exit(0)

if not a.ckpt:
    sys.exit("score mode needs --ckpt")

from distill import checkpoint

baseline = retention.load_baseline(out)
student = EdgeStudent(cfg).cuda().eval()

# Weight drift wants the pristine tables, so snapshot them before loading the
# trained weights in; the trained state dict is already LoRA-merged.
base_sd = {k: v.detach().to("cpu", copy=True) for k, v in student.model.state_dict().items()}

traj_spec = torch.load(Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt",
                       weights_only=False)  # pickled callables, D-032
student.extend_trajectory_vocab(traj_spec)
checkpoint.load_into(student, a.ckpt)

text = retention.text_drift(student, baseline)
weights = retention.weight_drift(base_sd, student.model.state_dict())
print(retention.format_report(text, weights))

report = Path(a.ckpt) / "retention.json"
report.write_text(json.dumps({"text": text, "weights": weights}, indent=2))
print(f"\nwrote {report}")
