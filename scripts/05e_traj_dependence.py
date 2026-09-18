"""Where does the teacher-forced trajectory get its information? (D-044 follow-up)

Run 7b's epoch-0 checkpoint decodes a plan that does not change when the CoC
changes (05d: end-speed gap 0.00 m/s), yet its teacher-forced GT loss is low.
Under teacher forcing the model sees four things at every trajectory position:
the frames, the CoC text, the ego history + route, and the trajectory PREFIX.
This scores the same windows with each of them degraded and reports the soft
GT cross-entropy per position group, so the answer is a number per source
rather than a guess.

Conditions (teacher-forced, clean GT prefix unless stated):
    full            frames + cached CoC + clean prefix
    no_img          frames zeroed (exactly what image dropout serves in training)
    swap_coc        CoC replaced by a WRONG maneuver (stop <-> accelerate)
    no_img+swap     both
    jitter16        prefix jittered at sigma 16 (run 7b's training maximum)
    jitter64        prefix jittered at sigma 64
    no_hist         the 48 ego-history slots hold the placeholder id (no ego motion)
    no_hist+no_img  history blanked AND frames zeroed: prefix + CoC + route only

If `swap_coc` costs nothing at the first positions, the CoC is not read even
where the prefix cannot yet reveal the plan. If `no_img` costs little, the
scene is not read either, and the prefix + history are doing the work.

    python scripts/05e_traj_dependence.py --ckpt <dir> --n 40
"""
from __future__ import annotations
import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

from distill.config import load_config                              # noqa: E402
from distill import checkpoint, losses                              # noqa: E402
from distill.data import grounding                                  # noqa: E402
from distill.data.dataset import Stage1Dataset, collate_stage1, move_batch  # noqa: E402
from distill.data.splits import load_split                          # noqa: E402
from distill.eval.coc_score import parse                            # noqa: E402
from distill.student.edge_wrapper import EdgeStudent                # noqa: E402

log = logging.getLogger("traj_dependence")

STOP_COC = "Stop for the red traffic light since the signal is red"
GO_COC = "Accelerate to proceed since the road ahead is clear"
FIRST = 8      # first 4 waypoints (curv, acc interleaved)


def swap_coc(text: str) -> str:
    m = parse(text).get("maneuver")
    return GO_COC if m in ("STOP", "SLOW", "YIELD") else STOP_COC


def jitter(bins: list[int], sigma: float, g: torch.Generator) -> list[int]:
    off = torch.round(torch.randn(len(bins), generator=g) * sigma).to(torch.long)
    return [int(b) for b in (torch.as_tensor(bins) + off).clamp_(0, 2999)]


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--adapters", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(a.config)
    sigma_t = float(cfg.stage1.get("gt_soft_sigma_bins", 6.0))

    student = EdgeStudent(cfg).cuda()
    student.extend_trajectory_vocab(torch.load(
        Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt", weights_only=False))
    student.lm._ensure_vision_tower()
    meta = checkpoint.load_into(student, Path(a.ckpt))
    if a.adapters:
        student.param_groups_stage1()
        checkpoint.load_adapters(student, a.adapters)
    log.info("checkpoint: %s (epoch=%s)", a.ckpt, meta.get("epoch"))
    student.eval()

    ds = Stage1Dataset(cfg, student.context_builder(), clip_ids=load_split(cfg, a.split))
    pad_id = student.tokenizer.pad_token_id
    lo, hi = student.future_base, student.future_base + student.n_future_bins
    g = torch.Generator().manual_seed(0)
    n = min(a.n, len(ds))
    acc = defaultdict(list)          # condition -> list of (128,) per-position CE

    def score(item, ctx, img_drop: bool) -> np.ndarray:
        it = {**item, "student": ctx, "img_drop": img_drop}
        batch = move_batch(collate_stage1([it], pad_id))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = student.ar_forward(batch, capture_layers=())
            tl, _, ok = losses.gather_targets(out["logits"], batch["input_ids"],
                                              batch["traj_pos"], 128)
            ce = losses.gt_traj_soft_ce(tl, batch["gt_traj_tok"] + lo,
                                        batch["traj_mask"] & ok, sigma_t, lo, hi,
                                        per_pos=True)
        return ce[0].float().cpu().numpy()

    for i in range(n):
        item = ds[i]
        path = ds.shards[i]
        window = ds._window(path.parent.name, int(path.stem))
        gt_bins = [int(b) for b in item["gt_traj_token_ids"]]
        coc = str(item["coc_text"])
        hint = grounding.route_hint(item["gt_future_xyz"]) if ds.route_hint else None
        build = lambda text, bins: ds.ctx.build(window, coc_text=text, traj_bins=bins,  # noqa: E731
                                                nav_text=hint)
        full = item["student"]
        acc["full"].append(score(item, full, False))
        acc["no_img"].append(score(item, full, True))
        sw = build(swap_coc(coc), gt_bins)
        acc["swap_coc"].append(score(item, sw, False))
        acc["no_img+swap"].append(score(item, sw, True))
        acc["jitter16"].append(score(item, build(coc, jitter(gt_bins, 16.0, g)), False))
        acc["jitter64"].append(score(item, build(coc, jitter(gt_bins, 64.0, g)), False))
        # Ego history: the same 48 slots, holding the placeholder the teacher
        # emits BEFORE replace_pad_token fills them (D-029) - no motion at all.
        hs, he = full["history_span"]
        nh = {**full, "input_ids": full["input_ids"].clone()}
        nh["input_ids"][hs:he] = student.special_ids["<|traj_history|>"]
        acc["no_hist"].append(score(item, nh, False))
        acc["no_hist+no_img"].append(score(item, nh, True))
        if (i + 1) % 10 == 0:
            log.info("  %d/%d", i + 1, n)

    log.info("\n%s  windows=%d  soft GT CE (nats; floor ~%.2f)", Path(a.ckpt).parent.name, n,
             float(np.log(sigma_t * np.sqrt(2 * np.pi * np.e))))
    log.info("  %-12s %7s %9s %9s %8s %8s", "condition", "all", f"pos<{FIRST}", f"pos>={FIRST}",
             "curv", "accel")
    for k, rows in acc.items():
        m = np.stack(rows)                     # (n, 128)
        log.info("  %-12s %7.3f %9.3f %9.3f %8.3f %8.3f", k, m.mean(), m[:, :FIRST].mean(),
                 m[:, FIRST:].mean(), m[:, 0::2].mean(), m[:, 1::2].mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
