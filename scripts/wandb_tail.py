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
    r"epoch (\d+) step (\d+)/(\d+) loss ([\d.]+) \[([^\]]*)\]")
#: `<term> <value>` pairs inside the breakdown bracket. Parsed by NAME, not by
#: capture-group position: run 4 added `traj_accel` and `gt_soft` to the
#: breakdown, and the positional regex this replaced would have silently
#: mislabelled every series after `traj` rather than failing.
TERM_RE = re.compile(r"(\w+) (-?[\d.]+)")
#: breakdown term -> W&B metric. `gt` keeps `loss/gt_ce` so the one-hot GT curve
#: stays continuous with runs 1-3; run 4's trained term is `loss/gt_ce_soft`.
TERM_METRIC = {"traj": "loss/traj_kl", "traj_accel": "loss/traj_kl_accel",
               "text": "loss/text_kl", "struct": "loss/struct_ce",
               "feat": "loss/feat", "gt_soft": "loss/gt_ce_soft",
               "gt": "loss/gt_ce"}
GATE_RE = re.compile(
    r"epoch (\d+) challenging coarse-minADE ([\d.]+) m \| val CoC NLL ([\d.]+) "
    r"struct NLL ([\d.]+)(?: \| gt_ce_w ([\d.]+))?")
# `no .*?improvement` because run 5 names the metric in the message
# ("early stop: no coc_nll improvement..."); the bare run-1..4 wording
# still matches, so old logs replay unchanged.
DONE_RE = re.compile(r"early stop: no .*?improvement|^Finished |labeling done")


class _DryRun:
    """Stand-in for a wandb run: prints what would be logged. --dry-run only."""

    def log(self, payload, step=None):
        print(f"step {step}: " + "  ".join(
            f"{k}={v}" for k, v in sorted(payload.items())))

    def alert(self, **kw):
        pass

    def finish(self):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--project", default="alpamayo2edge")
    ap.add_argument("--name", required=True, help="W&B run name AND resume id")
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse and print, create no W&B run. Use to check the "
                         "log format still parses before committing a long job "
                         "to it - the metric names here are derived from the "
                         "trainer's breakdown line and silently follow it.")
    a = ap.parse_args()

    if a.dry_run:
        run = _DryRun()
        a.once = True
    else:
        run = wandb.init(project=a.project, name=a.name, id=a.name, resume="allow")
    if not a.once:
        # The sbatch sidecar boots this right before the training call, so the
        # alert lands when the SLURM job actually dispatches (mail is disabled
        # cluster-side; --mail-type is a no-op). Delivery is opt-in per wandb
        # account: Settings -> Alerts -> email/Slack. Never let it stop the tail.
        try:
            run.alert(title=f"{a.name} started",
                      text=f"stage-1 training dispatched; tailing {a.log.name}")
        except Exception as e:                                          # noqa: BLE001
            print(f"wandb_tail: start alert failed ({e})")
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
                payload = {"epoch": int(m.group(1)),
                           "loss/total": float(m.group(4))}
                for term, val in TERM_RE.findall(m.group(5)):
                    payload[TERM_METRIC.get(term, f"loss/{term}")] = float(val)
                run.log(payload, step=step)
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

    if done and not a.once:
        try:
            run.alert(title=f"{a.name} finished",
                      text="log shows early stop / Finished - check the gate curve")
        except Exception as e:                                          # noqa: BLE001
            print(f"wandb_tail: finish alert failed ({e})")
    run.finish()


if __name__ == "__main__":
    main()
