"""Phase-2 smoke: the flow head's first forward, backward, and sample.

Loads the phase-2 init (stage-1 merged + RL adapters folded), builds the flow
context for a few cached PhysicalAI-AV val windows, runs one training step on
the gen tower (LoRA or full FT per config / --full-ft), then draws k samples
with the 10-step ODE, converts them to xyz through the teacher's action space
and prints minADE against the driver. The domain-31 head rows are untrained
here, so the number is a shape/finiteness check, NOT a result. Also draws the
SDE path once for the log-prob plumbing.

    bash scripts/03d_smoke_flow.sh                # Ada, LoRA arm, micro-batch 2
    bash scripts/03d_smoke_flow.sh --full-ft      # needs the Blackwell (~50 GB)
"""
from __future__ import annotations
import argparse
import functools
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
import torch                                                        # noqa: E402
from torch.utils.data import DataLoader, Subset                     # noqa: E402

from distill.config import load_config                              # noqa: E402
from distill import losses, train_stage2 as ts2                     # noqa: E402
from distill.data.dataset import collate_stage2, move_batch         # noqa: E402
from distill.optim import build_optimizer, trainable_report         # noqa: E402

log = logging.getLogger("smoke_flow")


def action_space_from_spec(cfg):
    from alpamayo1_5.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace
    spec = torch.load(Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt", weights_only=False)
    kw = {k: v for k, v in spec["binning"]["action_space_cfg"].items() if not k.startswith("_")}
    return UnicycleAccelCurvatureActionSpace(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-windows", type=int, default=2)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--full-ft", action="store_true", help="force student.lora.stage2.enabled=false")
    ap.add_argument("--lora", action="store_true", help="force student.lora.stage2.enabled=true")
    ap.add_argument("--no-adapters", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(a.config)
    if a.full_ft:
        cfg.student.lora.stage2.raw["enabled"] = False
    if a.lora:
        cfg.student.lora.stage2.raw["enabled"] = True
    if a.no_adapters:
        cfg.stage2.raw["init_adapters"] = None
    cfg.stage2.raw["dataset"] = "physicalai"
    cfg.stage2.raw["supervision"] = "gt_flow"      # no teacher tuples needed; cached ones also checked below

    t0 = time.time()
    student = ts2.load_student(cfg)
    log.info("student loaded in %.0f s, %.1f GiB resident", time.time() - t0,
             torch.cuda.memory_allocated() / 2**30)
    groups = student.param_groups_stage2()
    log.info("LoRA: %s | %s", student.lora_stats, trainable_report(groups))
    opt = build_optimizer(groups, cfg.stage2.lr)

    ds, collate = ts2.build_dataset(cfg, student, "val")
    ds = Subset(ds, range(a.n_windows))
    dl = DataLoader(ds, batch_size=a.n_windows, shuffle=False, num_workers=0,
                    collate_fn=functools.partial(collate, pad_id=student.tokenizer.pad_token_id))
    batch = move_batch(next(iter(dl)))
    L = int(batch["attention_mask"].sum(1).max())
    log.info("batch: %d windows, context up to %d tokens, cached flow tuples: %s",
             batch["input_ids"].shape[0], L, "flow_t" in batch)

    # ---- forward / backward -------------------------------------------------
    student.train()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ctx = student.build_flow_context(batch)
        kept = ctx.key_mask.sum(1).tolist()
        log.info("flow context: kept %s of %s tokens (future bins masked), next_pos %s, "
                 "K/V %s x %s", kept, batch["attention_mask"].sum(1).tolist(),
                 ctx.next_pos.tolist(), len(ctx.keys), tuple(ctx.keys[0].shape))
        w = losses.stage_weights(cfg.stage2, 0, 1)
        loss, parts = ts2.stage2_loss(student, batch, cfg, w, ctx)
        if "flow_t" in batch:                         # the D-007 path, on the cached tuples
            with torch.no_grad():
                v_pred = student.flow_forward(batch["flow_a_t"], batch["flow_t"], ctx, batch["flow_owner"])
            parts["flow_distill_cached"] = float(losses.flow_distill(v_pred, batch["flow_v"]))
    assert torch.isfinite(loss), f"non-finite loss {loss}"
    loss.backward()
    gen_grad = sum(float(p.grad.abs().sum()) for p in student.trainable_parameters() if p.grad is not None)
    ar_grad = [n for n, p in student.lm.model.named_parameters()
               if p.grad is not None and not student._is_diff(n) and "lora_" not in n
               and float(p.grad.abs().sum()) > 0]
    log.info("loss %.4f %s | trainable grad L1 %.3e | AR-tower params with grad: %d",
             float(loss), parts, gen_grad, len(ar_grad))
    assert gen_grad > 0, "no gradient reached the gen tower"
    assert not ar_grad, f"AR tower received gradients: {ar_grad[:3]}"
    torch.nn.utils.clip_grad_norm_(student.trainable_parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    log.info("train step OK  peak %.1f GiB", torch.cuda.max_memory_allocated() / 2**30)

    # ---- sample ---------------------------------------------------------------
    student.eval()
    space = action_space_from_spec(cfg)
    B = batch["gt_traj"].shape[0]
    owner = torch.arange(B, device="cuda").repeat_interleave(a.k)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ctx = student.build_flow_context(batch)
        out = student.sample_actions(ctx, owner, steps=a.steps, temperature=0.6)
        sde = student.sample_actions(ctx, torch.arange(B, device="cuda"), steps=a.steps, sde_noise=0.3)
    dt = time.time() - t1
    acts = out["actions"].float().cpu().view(B, a.k, 64, 2)
    assert torch.isfinite(acts).all(), "non-finite actions"
    hist_xyz, hist_rot = batch["hist_xyz"].float().cpu(), batch["hist_rot"].float().cpu()
    ades = []
    for b in range(B):
        xyz, _ = space.action_to_traj(acts[b], hist_xyz[b].expand(a.k, -1, -1),
                                      hist_rot[b].expand(a.k, -1, -1, -1))
        gt = batch["gt_future_xyz"][b].float().cpu()
        ade = (xyz[..., :2] - gt[None, :, :2]).norm(dim=-1).mean(-1)     # (k,)
        ades.append(float(ade.min()))
        log.info("window %d: sampled xyz end %s | gt end %s | minADE_%d %.2f m", b,
                 xyz[0, -1, :2].tolist(), gt[-1, :2].tolist(), a.k, ades[-1])
    log.info("sampler: %d windows x %d samples x %d steps in %.1f s, peak %.1f GiB; "
             "SDE logp %s", B, a.k, a.steps, dt, torch.cuda.max_memory_allocated() / 2**30,
             [round(x, 1) for x in sde["logp"].tolist()])
    log.info("SMOKE OK  mean minADE_%d (UNTRAINED head, sanity only) %.2f m", a.k,
             sum(ades) / len(ades))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
