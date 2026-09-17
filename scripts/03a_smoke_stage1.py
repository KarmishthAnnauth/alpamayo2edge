"""One batch through the entire stage-1 path, with every shape printed.

Stage 1 has never executed. Between the cache and the optimizer step sit the
image processor, `prepare_multimodal_reasoner_inputs`, the extended vocabulary,
`reasoner_forward` + per-layer capture, four losses and a LoRA backward — and
every one of them is written-but-unrun against real weights. Discovering that in
hour six of a training run is the expensive way.

    python scripts/03a_smoke_stage1.py                 # forward + backward, 1 batch
    python scripts/03a_smoke_stage1.py --gate 2        # ...then 2 windows of the gate
    python scripts/03a_smoke_stage1.py --micro-batch 1 # if 4 does not fit

Exit codes: 0 everything ran, 1 something failed (the traceback is the message).

What a PASS actually establishes — these are the open VALIDATE-ON-GPU items, and
each is checked here rather than asserted:

  * which image processor the Cosmos3-Edge snapshot ships, and whether its
    `pixel_values`/`image_grid_thw` satisfy `prepare_multimodal_reasoner_inputs`
    (D-029);
  * that the placeholder count per frame agrees with the assembled context — the
    `ContextBuilder` raises if it does not;
  * D_t, the teacher hidden size the feature cache was written at, which A1.5's
    config.json does not carry (D-019);
  * that `action_domain_id` is a free embodiment slot (D-017);
  * that the four loss terms are finite and of comparable magnitude at their
    configured weights, which is what decides whether `feat: 0.5` is a
    contribution or a rounding error.
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, "src")
import torch                                                        # noqa: E402

from distill.config import load_config                              # noqa: E402
from distill import losses                                          # noqa: E402
from distill.data.dataset import Stage1Dataset, collate_stage1, move_batch  # noqa: E402
from distill.data.splits import load_split                          # noqa: E402
from distill.student.edge_wrapper import EdgeStudent                # noqa: E402
from distill.student.layer_map import (FeatureProjections, pool_prompt_segments,
                                       uniform_map)                 # noqa: E402

log = logging.getLogger("smoke")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--gate", type=int, default=0,
                    help="also run the epoch gate over N windows (0 = skip)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(a.config)

    print("== load ==")
    student = EdgeStudent(cfg).cuda()
    spec_path = Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt"
    # This unpickles the teacher tokenizer's bound `decode`, so alpamayo1.5 has to
    # be importable in THIS env, not only in the labeling one (D-032).
    traj_spec = torch.load(spec_path, weights_only=False)
    student.extend_trajectory_vocab(traj_spec)
    print(f"  vocab {student.new_token_range}  future_base {student.future_base}  "
          f"hist_base {student.hist_base}")
    n_domains = getattr(student.net.config, "num_embodiment_domains", "?")
    print(f"  action_domain_id {student.action_domain_id} of {n_domains} slots")

    # This is what INJECTS the adapters and unfreezes the appended rows — without
    # it nothing has a gradient and the memory figure below is meaningless.
    groups = student.param_groups_stage1()
    print(f"  lora {student.lora_stats}")

    feat_layers = [int(l) for l in cfg.teacher.raw["feat_layers"]]
    pool_len = int(cfg.teacher.get("feat_pool_len", 8))
    n_student_layers = student.n_layers

    print("== data ==")
    # The train split's own settings, minus the grounding filter (a two-minute
    # scan the smoke does not need). Image dropout is forced on row 0 rather
    # than drawn, so the dropped path is exercised every time (D-043).
    ds = Stage1Dataset(cfg, student.context_builder(), clip_ids=load_split(cfg, "train"),
                       image_dropout=float(cfg.stage1.get("image_dropout", 0.0)))
    mb = int(a.micro_batch or cfg.stage1.micro_batch)
    items = [ds[i] for i in range(min(mb, len(ds)))]
    if float(cfg.stage1.get("image_dropout", 0.0)) > 0:
        items[0]["img_drop"] = True
        if len(items) > 1:
            items[1]["img_drop"] = False
    batch = move_batch(collate_stage1(items, pad_id=student.tokenizer.pad_token_id))
    print(f"  shards {len(ds)}  micro_batch {mb}  traj_prefix {ds.traj_prefix}  "
          f"route_hint {ds.route_hint}  img_drop {batch['img_drop'].tolist()}")
    print(f"  input_ids {tuple(batch['input_ids'].shape)}  "
          f"n_prompt {batch['n_prompt'].tolist()}")
    # The context carries what the config says it carries.
    if ds.route_hint:
        rs = student.special_ids["<|route_start|>"]
        n_route = int((batch["input_ids"] == rs).sum(1).min())
        print(f"  route hints: {[it['route_hint'] for it in items]}")
        if n_route != 1:
            raise RuntimeError("stage1.route_hint is on but a context has no "
                               "<|route_start|> token")
    if ds.traj_prefix == "gt":
        gt_ids = batch["gt_traj_tok"] + student.future_base
        forced = batch["input_ids"][batch["traj_pos"]].view(gt_ids.shape[0], -1)
        if not torch.equal(forced, gt_ids):
            raise RuntimeError("traj_prefix is 'gt' but the teacher-forced trajectory "
                               "positions do not hold the GT bins")
        print("  teacher-forced trajectory prefix == GT bins: ok")
    if bool(batch["img_drop"][0]):
        n_img = batch["image_grid_thw"].prod(-1)
        n0 = int(n_img[: int(items[0]["student"]["image_grid_thw"].shape[0])].sum())
        if float(batch["pixel_values"][:n0].abs().max()) != 0.0:
            raise RuntimeError("row 0 is marked img_drop but its pixel values are not zero")
        if bool(batch["coc_pos"][0].any()) or bool(batch["struct_pos"][0].any()):
            raise RuntimeError("row 0 is marked img_drop but still carries CoC/struct targets")
        if not bool(batch["traj_pos"][0].any()):
            raise RuntimeError("row 0 is marked img_drop and lost its trajectory targets")
        print(f"  image dropout on row 0: {n0} zeroed patches, CoC/struct masked, "
              "trajectory kept: ok")
    if "pixel_values" in batch:
        print(f"  pixel_values {tuple(batch['pixel_values'].shape)}  "
              f"image_grid_thw {tuple(batch['image_grid_thw'].shape)}")
    d_t = batch["feats"][feat_layers[0]].shape[-1]
    print(f"  teacher feats: {len(feat_layers)} layers x {pool_len} segments x D_t={d_t}"
          f"   <- D-019's open number")
    print(f"  traj {tuple(batch['traj'].shape)}  coc {tuple(batch['coc'].shape)}  "
          f"topk {tuple(batch['topk_idx'].shape)}")

    print("== forward ==")
    lmap = uniform_map(feat_layers, n_student_layers)   # smoke test: skip the CKA probe
    projections = FeatureProjections(
        lmap, student.hidden_size, d_t).cuda()
    if cfg.stage1.grad_checkpoint:
        student.enable_gradient_checkpointing()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = student.ar_forward(batch, capture_layers=lmap.keys())
        print(f"  logits {tuple(out['logits'].shape)}  captured "
              f"{len(out['hidden'])} layers")
        proj = projections({int(k): pool_prompt_segments(v, batch["n_prompt"], pool_len)
                            for k, v in out["hidden"].items()})
        n_traj = batch["traj"].shape[1]
        traj_logits, _, traj_ok = losses.gather_targets(
            out["logits"], batch["input_ids"], batch["traj_pos"], n_traj)
        n_coc = int(batch["coc_pos"].sum(1).max())
        coc_logits, coc_tgt, coc_ok = losses.gather_targets(
            out["logits"], batch["input_ids"], batch["coc_pos"], n_coc)
        n_struct = int(batch["struct_pos"].sum(1).max())
        struct_logits, struct_tgt, struct_ok = losses.gather_targets(
            out["logits"], batch["input_ids"], batch["struct_pos"], n_struct)
        traj_mask = batch["traj_mask"] & traj_ok
        print(f"  target positions: traj {int(traj_mask.sum())}/{traj_mask.numel()}  "
              f"coc {int(coc_ok.sum())} student tokens vs "
              f"{int(batch['coc_mask'].sum())} teacher tokens (cross-family, D-011)  "
              f"struct {int(struct_ok.sum())} (cot_end + traj_future_start subwords)")
        if int(struct_ok.sum()) == 0 and not bool(batch["img_drop"].all()):
            raise RuntimeError(
                "no structural target positions — struct_span never landed; the "
                "CoC terminator would be unsupervised, which is the run-1 bug")
        if int(traj_mask.sum()) == 0:
            raise RuntimeError(
                "no trajectory target positions — traj_span never landed in the "
                "context; the losses would be taken over an empty mask and read 0")

        # Mirrors train_stage1's terms block exactly — including run 4's per-dim
        # split of the teacher KL and the soft GT anchor. If the two drift apart
        # this stops being a pre-flight check and starts validating a loss the
        # 48h job will not run. See train_stage1.py for why the split exists.
        dim0 = torch.zeros_like(traj_mask)
        dim0[:, 0::2] = True                     # even = curvature, odd = accel
        curv_mask, acc_mask = traj_mask & dim0, traj_mask & ~dim0
        topk_idx = batch["topk_idx"] + student.future_base
        gt_bins = batch["gt_traj_tok"] + student.future_base
        w = losses.stage_weights(cfg.stage1, step=0, total_steps=1000)
        kl_on = bool(w.get("traj_kl", 0.0) or w.get("traj_kl_accel", 0.0))
        with torch.enable_grad() if kl_on else torch.no_grad():   # as the trainer does
            traj_kl_pos = losses.traj_topk_kl_per_pos(traj_logits, topk_idx,
                                                      batch["topk_logp"])
        terms = {
            "traj_kl": losses.masked_mean(traj_kl_pos, curv_mask),
            "traj_kl_accel": losses.masked_mean(traj_kl_pos, acc_mask),
            "text_kl": losses.text_kl_or_ce(coc_logits, coc_tgt, coc_ok, vocab_ok=True),
            "struct_ce": losses.text_kl_or_ce(struct_logits, struct_tgt, struct_ok, vocab_ok=True),
            "feat": losses.feature_match(proj, batch["feats"]),
            "gt_ce": losses.gt_traj_soft_ce(
                traj_logits, gt_bins, traj_mask,
                sigma_bins=float(cfg.stage1.get("gt_soft_sigma_bins", 6.0)),
                lo=student.future_base,
                hi=student.future_base + student.n_future_bins),
        }
        if int(curv_mask.sum()) == 0 or int(acc_mask.sum()) == 0:
            raise RuntimeError(
                f"per-dim split is degenerate: {int(curv_mask.sum())} curvature / "
                f"{int(acc_mask.sum())} accel positions. The future stream must "
                "interleave the two action dims (D-031); one empty half means "
                "traj_kl_accel would silently weight nothing")
        print("  loss term        raw        weight   contribution")
        for k, v in terms.items():
            print(f"    {k:<12} {float(v):9.4f}   {w[k]:5.2f}   {w[k] * float(v):9.4f}")
            if not torch.isfinite(v):
                raise RuntimeError(f"{k} is not finite")
        loss = sum(w[k] * v for k, v in terms.items())
        # Diagnostic only, exactly as the trainer logs it: the run-1..3 one-hot
        # CE, for curve comparability. `gt_ce` above floors at the soft target's
        # own entropy (~3.21 nats at sigma 6), NOT at 0 — see gt_traj_soft_ce.
        with torch.no_grad():
            print(f"    {'gt_ce (1hot)':<12} "
                  f"{float(losses.gt_traj_ce(traj_logits, gt_bins, traj_mask)):9.4f}"
                  "       —   diagnostic, not summed")

    print("== backward ==")
    loss.backward()
    params = student.trainable_parameters() + list(projections.parameters())
    n_group = sum(len(g["params"]) for g in groups)
    print(f"  optimizer would see {n_group} tensors in {len(groups)} groups")
    n_grad = sum(1 for p in params if p.grad is not None)
    gnorm = torch.nn.utils.clip_grad_norm_(params, 1e9)
    print(f"  {n_grad}/{len(params)} trainable tensors have gradients  "
          f"|g| {float(gnorm):.4f}")
    if n_grad == 0:
        raise RuntimeError("nothing received a gradient — check the LoRA injection "
                           "scope and the row masks")
    print(f"  peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
          f"at micro_batch {mb}")

    if a.gate:
        print("== gate ==")
        from distill.eval.coarse_minade import coarse_minade
        student.zero_grad(set_to_none=True)
        score = coarse_minade(cfg, student, split="challenging",
                              k=int(cfg.eval.minade_k), max_windows=a.gate)
        print(f"  UNTRAINED coarse minADE_{cfg.eval.minade_k}: {score:.2f} m")
        print("  (a number, not a verdict — but it is the same metric epoch 1 "
              "will be compared against, and it should not be NaN or ~0)")

    print("\nOK — the stage-1 path runs end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
