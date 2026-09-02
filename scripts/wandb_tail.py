"""Stream an already-running stage-1 training log to Weights & Biases.

A sidecar: it does not touch the training job, just tails its log file and
re-emits the numbers. Use it to watch a run remotely (wandb.ai / the W&B app)
when train_stage1.py itself has no wandb wiring.

    wandb login                                  # one-time, paste key from wandb.ai/authorize
    python scripts/wandb_tail.py logs/a2e-stage1-207.out --name stage1-job207

`scripts/03_train_stage1.sh` launches this automatically in the background for
its own job, so a submitted run is watchable without anyone starting it by hand.
Run it manually only for an old log or to re-sync a run.

Resumable: re-running with the same --name re-scans from the top and de-dupes by
step, so a killed sidecar loses nothing. Exits when the log shows the run is
done (early stop / Finished) or --once is passed.

Metrics: loss/{total,traj_kl,text_kl,struct_ce,feat,gt_ce} per step (all
unweighted), gate/{coarse_minADE_m,val_coc_nll,val_struct_nll} per epoch, and
weights/gt_ce per epoch when the log carries the anneal schedule (run 3+).
"""
from __future__ import annotations
import argparse
import re
import time
from pathlib import Path

import wandb

STEP_RE = re.compile(
    r"epoch (\d+) step (\d+)/(\d+) loss ([\d.]+) \[traj ([\d.]+) text ([\d.]+) "
    r"struct ([\d.]+) feat ([\d.]+) gt ([\d.]+)\]")
GATE_RE = re.compile(
    r"epoch (\d+) challenging coarse-minADE ([\d.]+) m \| val CoC NLL ([\d.]+) "
    r"struct NLL ([\d.]+)(?: \| gt_ce_w ([\d.]+))?")
DONE_RE = re.compile(r"early stop: no improvement|^Finished |labeling done")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--project", default="alpamayo2edge")
    ap.add_argument("--name", required=True, help="W&B run name AND resume id")
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    a = ap.parse_args()

    run = wandb.init(project=a.project, name=a.name, id=a.name, resume="allow")
    seen: set[int] = set()
    last_step = 0
    done = False

    while True:
        text = a.log.read_text(errors="replace") if a.log.exists() else ""
        for line in text.splitlines():
            m = STEP_RE.search(line)
            if m:
                step = int(m.group(2))
                if step in seen:
                    continue
                seen.add(step)
                last_step = max(last_step, step)
                run.log({
                    "epoch": int(m.group(1)),
                    "loss/total": float(m.group(4)),
                    "loss/traj_kl": float(m.group(5)),
                    "loss/text_kl": float(m.group(6)),
                    "loss/struct_ce": float(m.group(7)),
                    "loss/feat": float(m.group(8)),
                    "loss/gt_ce": float(m.group(9)),
                }, step=step)
                continue
            g = GATE_RE.search(line)
            if g:
                key = f"__gate_{g.group(1)}"
                if key in seen:  # reuse the set as a generic dedupe
                    continue
                seen.add(key)  # type: ignore[arg-type]
                payload = {
                    "gate/coarse_minADE_m": float(g.group(2)),
                    "gate/val_coc_nll": float(g.group(3)),
                    "gate/val_struct_nll": float(g.group(4)),
                    "gate/epoch": int(g.group(1)),
                }
                if g.group(5) is not None:      # gt_ce anneal schedule (run 3+)
                    payload["weights/gt_ce"] = float(g.group(5))
                run.log(payload, step=last_step)
            if DONE_RE.search(line):
                done = True

        if done or a.once:
            break
        time.sleep(a.poll)

    run.finish()


if __name__ == "__main__":
    main()
