"""Stage 2 / phase 2: train the flow head (Edge's gen tower) on expert trajectories.

Two data arms through one loop (config `stage2.dataset`):

* `physicalai` - the plan-v2 cache: per window, K cached `(t, a_t, u_teacher)`
  tuples from the teacher's action expert (D-007), loss = `flow_distill` (+ a
  `fm_gt` anchor). No teacher in memory.
* `bench2drive` - CARLA expert trajectories, no teacher tuples: the trainer draws
  its own `(t, a0)` per window (`stage2.k_samples`, `stage2.t_sampler`) and the
  loss is `flow_matching_gt`, the objective the teacher's own stage 2 uses.

The AR tower is frozen: `stage2.init_ckpt` (merged stage-1 weights) plus
`stage2.init_adapters` (the phase-1.5 RL adapters, folded in before anything is
trained) supplies the CoC the flow head conditions on. One reasoner pass per
window builds a `FlowContext` (per-layer K/V over prompt + history + route +
CoC, future bins masked - flow_path.py) that the window's K flow samples share.

The gen tower is FULLY fine-tuned by default (student.lora.stage2.enabled:
false, fp32 master weights), NVIDIA's own action post-training split; LoRA r16
is the comparison arm. Our embodiment row of action2llm/llm2action is
full-rank in both (D-017).
"""
from __future__ import annotations
import argparse
import functools
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import load_config
from .data.dataset import Stage1Dataset, collate_stage2, move_batch
from .data.splits import load_split
from .student.edge_wrapper import EdgeStudent
from .student import lora as lora_mod
from . import checkpoint, losses
from .optim import build_optimizer, cosine_lr, set_lr, trainable_report

log = logging.getLogger(__name__)


class _WandB:
    """Direct W&B wiring for stage 2 (user request 2026-09-29: know when a run
    starts and ends). One run per job, named `<run_name>-job<id>`; `wandb.alert`
    fires on start, on finish and on failure, which W&B forwards to email/Slack
    per the account's alert settings. Off when `stage2.wandb` is false or wandb
    is not importable; never allowed to kill training."""

    def __init__(self, cfg, enabled: bool, tag: str):
        import os
        self.run = None
        if not enabled:
            return
        try:
            import wandb
            job = os.environ.get("SLURM_JOB_ID", "manual")
            name = f"{cfg.stage2.get('run_name', 'stage2')}-{tag}job{job}"
            self.run = wandb.init(project=cfg.stage2.get("wandb_project", "alpamayo2edge"),
                                  name=name, id=name, resume="allow",
                                  config={"stage2": dict(cfg.stage2.raw),
                                          "lora_stage2": dict(cfg.student.lora.stage2.raw)})
            self.run.alert(title=f"stage 2 started: {name}",
                           text=f"{cfg.stage2.get('dataset')} / {cfg.stage2.get('supervision')} / "
                                f"epochs {cfg.stage2.epochs} / init {cfg.stage2.get('init_ckpt')}")
            log.info("wandb: %s", self.run.url)
        except Exception as e:                                  # noqa: BLE001
            log.warning("wandb disabled (%s)", e)
            self.run = None

    def log(self, data: dict, step: int | None = None):
        if self.run is not None:
            try:
                self.run.log(data, step=step)
            except Exception as e:                              # noqa: BLE001
                log.warning("wandb.log failed: %s", e)

    def finish(self, ok: bool, text: str = ""):
        if self.run is None:
            return
        try:
            self.run.alert(title=f"stage 2 {'finished' if ok else 'FAILED'}: {self.run.name}", text=text)
            self.run.finish(exit_code=0 if ok else 1)
        except Exception as e:                                  # noqa: BLE001
            log.warning("wandb.finish failed: %s", e)


def load_student(cfg, device: str = "cuda") -> EdgeStudent:
    """Student at the phase-2 init: stage-1 merged weights + RL adapters folded in."""
    student = EdgeStudent(cfg).to(device)
    traj_spec = torch.load(Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt",
                           weights_only=False)  # pickled callables, D-032
    student.extend_trajectory_vocab(traj_spec)
    student.lm._ensure_vision_tower()
    init = Path(cfg.stage2.get("init_ckpt") or (Path(cfg.paths.runs_root) / "stage1" / "best"))
    log.info("stage-1 checkpoint: %s -> %s", init, init.resolve())
    checkpoint.load_into(student, init)
    adapters = cfg.stage2.get("init_adapters")
    if adapters:
        student.param_groups_stage1()                    # inject, load, fold
        meta = checkpoint.load_adapters(student, adapters)
        n = lora_mod.merge_lora_(student._decoder_layers())
        log.info("folded %d adapter modules from %s (step=%s) into the AR tower",
                 n, adapters, meta.get("step"))
    return student


def build_dataset(cfg, student, split: str = "train"):
    """(dataset, collate_fn) for `stage2.dataset`."""
    kind = cfg.stage2.get("dataset", "physicalai")
    if kind == "physicalai":
        return (Stage1Dataset(cfg, student.context_builder(), clip_ids=load_split(cfg, split)),
                collate_stage2)
    if kind == "bench2drive":
        from .data import bench2drive as b2d
        root = Path(cfg.paths.b2d_cache_root)
        coc = cfg.stage2.get("coc_cache")
        if coc and not Path(coc).exists():
            raise FileNotFoundError(f"stage2.coc_cache {coc} is missing - run "
                                    "scripts/05j_b2d_coc.py generate, or set it to null")
        ds = b2d.Bench2DriveDataset(root, student.context_builder(),
                                    clip_ids=b2d.load_b2d_split(root, split),
                                    coc_lookup=b2d.coc_lookup_from_cache(coc),
                                    for_generation=True)      # context ends at <|traj_future_start|>
        log.info("bench2drive %s: %d windows, CoC cache %s", split, len(ds), coc or "NONE (free-running)")
        return ds, b2d.collate_b2d
    raise ValueError(f"stage2.dataset must be physicalai | bench2drive, got {kind!r}")


def build_bucket_sampler(cfg, ds):
    """`stage2.buckets` (bench2drive only): a WeightedRandomSampler that draws each
    bucket of `data/b2d_buckets.py` with its configured share, same epoch length.
    None when disabled, i.e. the uniform shuffle of runs 1-4."""
    bcfg = cfg.stage2.raw.get("buckets") or {}
    if not bcfg.get("enabled", False):
        return None
    if cfg.stage2.get("dataset", "physicalai") != "bench2drive":
        raise ValueError("stage2.buckets is defined for dataset: bench2drive only")
    from torch.utils.data import WeightedRandomSampler
    from .data import b2d_buckets as bk
    table = bk.load(Path(cfg.paths.b2d_cache_root), "train")
    keys = [bk.shard_key(p) for p in ds.shards]
    missing = [k for k in keys if k not in table]
    if missing:
        raise RuntimeError(f"{len(missing)} train windows have no bucket (e.g. {missing[0]}) - "
                           "the cache changed; rerun scripts/01e_b2d_buckets.py")
    buckets = [table[k] for k in keys]
    share = dict(bcfg["share"])
    log.info("bucketed sampling:\n%s", bk.table(buckets, share))
    w = torch.as_tensor(bk.sampling_weights(buckets, share), dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=len(ds), replacement=True)


def draw_timesteps(kind: str, n: int, k: int, device) -> torch.Tensor:
    """Teacher-convention t (data weight) per (window, sample), shape (n*k,)."""
    if kind == "stratified":                             # TeacherWrapper.stratified_timesteps
        u = torch.rand(n, k, device=device)
        t = 0.999 * (torch.arange(k, device=device)[None, :] + u) / k
    elif kind == "beta":                                 # the teacher's law (D-012): s ~ Beta(1.5, 1),
        # t = 0.999 (1 - s) -> biased to t ~ 0, the HIGH-NOISE end where the sampler
        # commits to a mode (Edge's waver + shift 5 is high-noise too). Stratified
        # through the inverse CDF (F(s) = s^1.5) to keep the k draws spread.
        u = (torch.arange(k, device=device)[None, :] + torch.rand(n, k, device=device)) / k
        t = 0.999 * (1.0 - u.pow(1.0 / 1.5))
    elif kind == "logitnormal":                          # Edge's train-time sigma law, t = 1 - sigma
        sigma = torch.sigmoid(torch.randn(n, k, device=device))
        t = (1.0 - sigma).clamp(0.0, 0.999)
    else:
        raise ValueError(f"stage2.t_sampler must be stratified | beta | logitnormal, got {kind!r}")
    return t.reshape(-1)


def flow_targets(batch, cfg, device) -> dict:
    """(t, a_t, owner[, v_teacher]) for the micro-batch - cached or drawn."""
    if "flow_t" in batch:
        return dict(t=batch["flow_t"], a_t=batch["flow_a_t"], owner=batch["flow_owner"],
                    v_teacher=batch["flow_v"])
    a1 = batch["gt_traj"]                                # (B, 64, 2), teacher action space
    B = a1.shape[0]
    k = int(cfg.stage2.get("k_samples", 8))
    t = draw_timesteps(cfg.stage2.get("t_sampler", "stratified"), B, k, device)
    owner = torch.arange(B, device=device).repeat_interleave(k)
    a0 = torch.randn(B * k, *a1.shape[1:], device=device)
    tt = t.view(-1, 1, 1)
    a_t = tt * a1[owner] + (1 - tt) * a0
    return dict(t=t, a_t=a_t, owner=owner, v_teacher=None)


_ACTION_SPACE: dict = {}


def _action_space(cfg):
    """The cache's action space (its own normalisation), loaded once."""
    from .data import bench2drive as b2d
    root = str(cfg.paths.b2d_cache_root)
    if root not in _ACTION_SPACE:
        _ACTION_SPACE[root] = b2d.load_cache_action_space(Path(root), Path(cfg.paths.teacher_repo))
    return _ACTION_SPACE[root]


def traj_space_loss(cfg, batch, a_t, t, v_pred, owner) -> torch.Tensor:
    """Trajectory-space term (run 4, 2026-10-01): mean xy displacement, in metres,
    of the DENOISED action estimate integrated through the cache's unicycle model.

    Why: position is accel integrated twice, so a constant 0.1 sigma accel bias
    (MSE 0.01, invisible next to an fm loss of 0.3-0.4) is 2.1 m of ADE on the
    val windows - the whole open-loop error. The flow-matching loss is white
    over the 64 steps; this term is the metric itself.

    a_t = t a1 + (1-t) a0 and v = a1 - a0 (D-012), so a1_hat = a_t + (1-t) v.
    Weighted by t^p (`stage2.traj_loss_t_power`): at low t the estimate is the
    posterior MEAN of a wide distribution, whose xy optimum differs from the
    action-space optimum flow matching needs; near t = 1 the two agree.
    """
    tt = t.view(-1, 1, 1).float()
    with torch.autocast("cuda", enabled=False):
        a1_hat = a_t.float() + (1 - tt) * v_pred.float()
        xyz, _ = _action_space(cfg).action_to_traj(
            a1_hat, batch["hist_xyz"].float()[owner], batch["hist_rot"].float()[owner])
        gt = batch["gt_future_xyz"].float()[owner]
        d = ((xyz[..., :2] - gt[..., :2]).pow(2).sum(-1) + 1e-6).sqrt().mean(-1)      # (N,) metres
        wt = tt.view(-1).pow(float(cfg.stage2.get("traj_loss_t_power", 1.0)))
        return (wt * d).sum() / wt.sum().clamp_min(1e-6)


def coc_preservation_loss(student, batch, ctx) -> torch.Tensor | None:
    """CE on the cached CoC tokens (+ the `<|cot_end|><|traj_future_start|>`
    boundary) from the SAME reasoner pass that builds the flow context.

    Why: run 3 trained the AR-tower LoRA with the flow loss alone and the CoC
    the tower writes broke (terminated 1.00 -> 0.73, GT-consistent 0.735 ->
    0.656). The cached CoC is the init's own output, so this is self-
    distillation: it holds the language behaviour while the flow loss moves the
    perception. The lm_head runs on the gathered predictor positions only
    (p - 1 predicts p, `losses.gather_targets`' shift rule).
    """
    h = ctx.final_hidden
    if h is None or not h.requires_grad:
        return None
    mask = batch["coc_pos"] | batch["struct_pos"]
    mask = mask.clone(); mask[:, 0] = False
    b_idx, p_idx = mask.nonzero(as_tuple=True)
    if b_idx.numel() == 0:
        return None
    logits = student.lm.lm_head(h[b_idx, p_idx - 1])
    return F.cross_entropy(logits.float(), batch["input_ids"][b_idx, p_idx])


def stage2_loss(student, batch, cfg, w, ctx) -> tuple[torch.Tensor, dict]:
    tg = flow_targets(batch, cfg, ctx.key_mask.device)
    v_pred = student.flow_forward(tg["a_t"], tg["t"], ctx, tg["owner"])
    a1 = batch["gt_traj"][tg["owner"]]
    # a_t = t a1 + (1-t) a0 (D-012), so a0 = (a_t - t a1) / (1 - t); t <= 0.999.
    t = tg["t"].view(-1, 1, 1)
    a0 = (tg["a_t"] - t * a1) / (1 - t).clamp_min(1e-3)
    fm_gt = losses.flow_matching_gt(v_pred, a0, a1)
    supervision = cfg.stage2.get("supervision", "teacher_flow")
    if supervision == "gt_flow" or tg["v_teacher"] is None:
        if supervision != "gt_flow":
            raise RuntimeError("supervision=teacher_flow but the batch carries no cached "
                               "teacher tuples - set stage2.supervision: gt_flow")
        total, parts = fm_gt, {"fm_gt": float(fm_gt)}
        if w.get("traj", 0.0) > 0 and "gt_future_xyz" in batch:
            tr = traj_space_loss(cfg, batch, tg["a_t"], tg["t"], v_pred, tg["owner"])
            total = total + w["traj"] * tr
            parts["traj_m"] = float(tr)
        if w.get("coc_ce", 0.0) > 0:
            ce = coc_preservation_loss(student, batch, ctx)
            if ce is not None:
                total = total + w["coc_ce"] * ce
                parts["coc_ce"] = float(ce)
        return total, parts
    fd = losses.flow_distill(v_pred, tg["v_teacher"])
    return w["flow_distill"] * fd + w["fm_gt"] * fm_gt, {"flow_distill": float(fd), "fm_gt": float(fm_gt)}


def main(cfg_path: str, smoke: int = 0):
    cfg = load_config(cfg_path)
    if smoke:
        # `--smoke N`: N micro-batches of one epoch, a 16-window gate, and the
        # checkpoint under runs/stage2/smoke - the bench2drive path end to end
        # on the real card before the multi-day run is submitted.
        cfg.stage2.raw.update(epochs=1, gate_windows=16, run_name="smoke", grad_accum=1)
    wb = _WandB(cfg, enabled=bool(cfg.stage2.get("wandb", True)) and not smoke, tag="")
    try:
        _train(cfg, smoke, wb)
    except BaseException as e:                                  # noqa: BLE001
        wb.finish(ok=False, text=f"{type(e).__name__}: {e}"[:500])
        raise
    wb.finish(ok=True, text="all epochs done; see runs/stage2/<run_name>/best")


def _check_disk(cfg, smoke: int) -> None:
    """Fail at start, not at the first save: every epoch writes a ~7.8 GB merged
    checkpoint. Run 4's first attempt (job 529) died 2 h 43 min in with ENOSPC on
    a root disk that had 4.5 GB free. The run dir may be a symlink to /bulk."""
    import shutil
    run_dir = Path(cfg.paths.runs_root) / "stage2" / cfg.stage2.get("run_name", "run")
    run_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(run_dir.resolve()).free / 2**30
    need = 8.0 * (1 if smoke else int(cfg.stage2.epochs))
    if free < need:
        raise RuntimeError(f"{run_dir.resolve()}: {free:.0f} GiB free, the run writes ~{need:.0f} GiB "
                           "of checkpoints - free space or symlink the run dir to /bulk")
    log.info("checkpoints -> %s (%.0f GiB free, ~%.0f needed)", run_dir.resolve(), free, need)


def _train(cfg, smoke: int, wb: "_WandB"):
    _check_disk(cfg, smoke)
    student = load_student(cfg)

    ds, collate = build_dataset(cfg, student, "train")
    pad_id = student.tokenizer.pad_token_id
    sampler = build_bucket_sampler(cfg, ds)
    dl = DataLoader(ds, batch_size=cfg.stage2.micro_batch, shuffle=sampler is None,
                    sampler=sampler, num_workers=4, pin_memory=True,
                    collate_fn=functools.partial(collate, pad_id=pad_id))

    groups = student.param_groups_stage2()
    opt = build_optimizer(groups, cfg.stage2.lr, cfg.stage2.get("weight_decay", 0.0))
    log.info("LoRA: %s", student.lora_stats)
    log.info("%s", trainable_report(groups))

    gate_dl = None
    if cfg.stage2.get("dataset", "physicalai") == "bench2drive":
        from .eval.flow_minade import build_val_loader
        gate_dl = build_val_loader(cfg, student, split="val", n=cfg.stage2.get("gate_windows", 160),
                                   batch=cfg.stage2.get("gate_batch", 8), coc=cfg.stage2.get("coc_cache"))
        log.info("gate: %d val windows, k=%d", len(gate_dl.dataset), cfg.stage2.get("gate_k", 6))
    best = float("inf")

    steps_per_epoch = math.ceil(len(dl) / cfg.stage2.grad_accum)
    total_steps = steps_per_epoch * cfg.stage2.epochs
    warmup = int(total_steps * cfg.stage2.get("warmup_frac", 0.0))
    physicalai = cfg.stage2.get("dataset", "physicalai") == "physicalai"
    step = 0
    student.train()
    for epoch in range(cfg.stage2.epochs):
        for i, batch in enumerate(dl):
            if smoke and i >= smoke:
                break
            batch = move_batch(batch)
            frac = step / max(total_steps, 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if physicalai and frac >= cfg.stage2.scheduled_sampling_start_frac:
                    with torch.no_grad():                # student-sampled prefix (D-007 plan)
                        batch["traj"] = student.generate_traj_tokens(batch)
                w = losses.stage_weights(cfg.stage2, step, total_steps)
                # reasoner once per window (graph kept iff AR LoRA); its final
                # hidden states only when the CoC-preservation term reads them
                ctx = student.build_flow_context(batch, keep_final_hidden=w.get("coc_ce", 0.0) > 0)
                loss, parts = stage2_loss(student, batch, cfg, w, ctx)
            (loss / cfg.stage2.grad_accum).backward()
            if (i + 1) % cfg.stage2.grad_accum == 0:
                set_lr(opt, cosine_lr(step, total_steps, warmup))
                torch.nn.utils.clip_grad_norm_(student.trainable_parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                step += 1
                wb.log({"loss/total": float(loss), **{f"loss/{k}": v for k, v in parts.items()},
                        "lr": opt.param_groups[0]["lr"], "epoch": epoch}, step=step)
                if step % 20 == 0 or smoke:
                    log.info("epoch %d step %d/%d loss %.4f %s  peak %.1f GiB", epoch, step,
                             total_steps, loss.item(), parts,
                             torch.cuda.max_memory_allocated() / 2**30)

        run_dir = Path(cfg.paths.runs_root) / "stage2" / cfg.stage2.get("run_name", "run")
        meta = {"stage": "stage2", "epoch": epoch, "step": step}
        if gate_dl is not None:
            from .eval.flow_minade import evaluate_flow, log_result
            res = evaluate_flow(cfg, student, gate_dl, k=cfg.stage2.get("gate_k", 6),
                                steps=cfg.stage2.get("sample_steps", 10))
            log_result(res, f"epoch {epoch} gate")
            gate_metric = cfg.stage2.get("gate_metric", "t0_ade")
            meta.update(val_minade=res["minade"], val_t0_ade=res["t0_ade"], val_cv_ade=res["cv_ade"],
                        val_n=res["n"], gate_metric=gate_metric)
            wb.log({"gate/t0_ade": res["t0_ade"], "gate/minade": res["minade"], "gate/meanade": res["meanade"],
                    "gate/minfde": res["minfde"], "gate/cv_ade": res["cv_ade"],
                    "gate/braked_minade": res["braked"]["minade"],
                    "gate/other_minade": res["not_braked"]["minade"], "epoch": epoch}, step=step)
        ckpt = run_dir / f"epoch{epoch}"
        checkpoint.save(student, ckpt, save_dtype=torch.bfloat16, **meta)
        log.info("saved %s", ckpt)
        if gate_dl is not None and res[gate_metric] < best:
            best = res[gate_metric]
            link = run_dir / "best"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(ckpt.name)
            log.info("best -> %s (val %s %.3f)", ckpt.name, gate_metric, best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", type=int, default=0, help="N micro-batches, then gate + save, and stop")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(a.config, smoke=a.smoke)
