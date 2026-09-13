"""Where the CoC cross-entropy lives, position by position.

`losses.text_kl_or_ce` is a per-token MEAN of hard-label CE over the teacher's
one sampled trace. A trace is ~14 tokens; the maneuver decision is the first one
or two, the rest is template ("... since it is directly ahead in our lane"). If
the template tokens are near-free, the loss the optimiser sees is dominated by
positions that carry no decision, and a student can post a tiny `text` loss
while hedging the maneuver - which is the mode-collapse pattern 05b measures.

Teacher-forced, one forward per batch, no generation. Prints mean CE per CoC
position and the share of total CE in the first k positions.

    python scripts/05c_coc_ce_by_position.py --ckpt <dir> --split val --n 200
"""
from __future__ import annotations
import argparse
import functools
import logging
import sys
from pathlib import Path

sys.path.insert(0, "src")
import torch                                                        # noqa: E402
import torch.nn.functional as F                                     # noqa: E402
from torch.utils.data import DataLoader, Subset                     # noqa: E402

from distill.config import load_config                              # noqa: E402
from distill import checkpoint, losses                              # noqa: E402
from distill.data.dataset import Stage1Dataset, collate_stage1, move_batch  # noqa: E402
from distill.data.splits import load_split                          # noqa: E402
from distill.student.edge_wrapper import EdgeStudent                # noqa: E402

log = logging.getLogger("coc_ce_pos")


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-pos", type=int, default=24)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(a.config)

    student = EdgeStudent(cfg).cuda()
    student.extend_trajectory_vocab(torch.load(
        Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt", weights_only=False))
    student.lm._ensure_vision_tower()
    ckpt = Path(a.ckpt or (Path(cfg.paths.runs_root) / "stage1" / "best"))
    meta = checkpoint.load_into(student, ckpt)
    log.info("checkpoint: %s (epoch=%s)", ckpt, meta.get("epoch"))
    student.eval()

    ds = Stage1Dataset(cfg, student.context_builder(), clip_ids=load_split(cfg, a.split))
    ds = Subset(ds, range(min(a.n, len(ds))))
    dl = DataLoader(ds, batch_size=cfg.stage1.micro_batch, shuffle=False, num_workers=2,
                    collate_fn=functools.partial(collate_stage1,
                                                 pad_id=student.tokenizer.pad_token_id))
    ce_sum = torch.zeros(a.max_pos, dtype=torch.float64)
    ce_cnt = torch.zeros(a.max_pos, dtype=torch.float64)
    tot_sum = tot_cnt = 0.0
    first_tok = {}
    for batch in dl:
        batch = move_batch(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = student.ar_forward(batch, capture_layers=())
            n_coc = int(batch["coc_pos"].sum(1).max())
            lg, tgt, ok = losses.gather_targets(out["logits"], batch["input_ids"],
                                                batch["coc_pos"], n_coc)
        ce = F.cross_entropy(lg.float().transpose(1, 2), tgt, reduction="none")  # (B, n)
        ce = ce * ok
        k = min(n_coc, a.max_pos)
        ce_sum[:k] += ce[:, :k].sum(0).cpu().double()
        ce_cnt[:k] += ok[:, :k].sum(0).cpu().double()
        tot_sum += float(ce.sum()); tot_cnt += float(ok.sum())
        # What the first CoC token is, and its CE - the maneuver verb lives here.
        for b in range(tgt.shape[0]):
            if ok[b, 0]:
                w = student.tokenizer.decode([int(tgt[b, 0])]).strip().lower()
                s = first_tok.setdefault(w, [0.0, 0])
                s[0] += float(ce[b, 0]); s[1] += 1

    mean_pos = (ce_sum / ce_cnt.clamp_min(1)).tolist()
    mean_all = tot_sum / max(tot_cnt, 1)
    log.info("\n%s split=%s epoch=%s  windows=%d  mean CE over all CoC tokens = %.3f",
             ckpt.name, a.split, meta.get("epoch"), len(ds), mean_all)
    log.info("  pos   mean CE   n")
    for i in range(a.max_pos):
        if ce_cnt[i] > 0:
            log.info("  %3d   %6.3f   %4d%s", i, mean_pos[i], int(ce_cnt[i]),
                     "   <- maneuver verb" if i == 0 else "")
    for k in (1, 2, 3, 5):
        share = float(ce_sum[:k].sum()) / max(tot_sum, 1e-9)
        log.info("  share of total CE in first %d position(s): %.1f%%  "
                 "(those are %.1f%% of tokens)", k, 100 * share,
                 100 * float(ce_cnt[:k].sum()) / max(tot_cnt, 1))
    log.info("  first-token CE by verb (teacher's word -> student's CE on it):")
    for w, (s, c) in sorted(first_tok.items(), key=lambda kv: -kv[1][1])[:10]:
        log.info("     %-12s n=%4d  mean CE %.3f", w, c, s / c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
