"""Phase 1.5: GRPO on the AR tower's chain-of-causation (D-037).

Why RL and not a better CE. `text_kl` is hard-label CE on ONE sampled teacher
trace per window, and 05c/05b measured what that produces: where the teacher
commits (nudge / adapt / lane change) the student holds ~0.1-0.25 on the
teacher's verb and the rest on "keep", so at T=0.6 it never says it - a
conditional that is under-sharpened, not mode-deficient (the mass reappears at
T=1.0, at the cost of accuracy). CE learns the expectation of a noisy target and
cannot sharpen it; group-relative reward can. It is also how the teacher got its
own CoC: Alpamayo 1.5 was RL post-trained with a reasoning reward on the
chain-of-causation (model card; recipes/alpamayo1_x_rl, GRPO, n_generation 12).

One process, one GPU, no vLLM: the student is 2B and a CoC is ~14 tokens.

  policy     run-314/best (merged) + FRESH LoRA adapters (zero-init B), so the
             policy starts exactly at the SFT checkpoint
  reference  the same weights with the adapters scaled to 0 - no second model
  rollouts   `generate_coc_text`, G samples per prompt at the teacher's own
             T=0.6 / top_p 0.98 (the setting the student is judged at)
  reward     `eval.coc_score` against the cached teacher trace: maneuver match,
             ego direction, object recall, a false-clear penalty, and a hard
             penalty for not terminating (rule-based, deterministic, free)
  advantage  group-normalised; zero-variance groups are skipped (the recipe's
             documented "GRPO group collapse")
  loss       per-token policy gradient + beta * KL(policy || reference)
             (k3 estimator) + ce_aux * the existing teacher-forced CE, as an
             anchor against reward-hacking the rule grader

Everything comes from the cache (student inputs + `coc_text`); no streaming.

Run 4 (`reward.mode: perspan`, D-040). The student's plan does not depend on
its CoC (forcing "turn left" vs "turn right" moves the decoded heading 2-6
deg), so run 3's one ADE-based advantage on both spans trained the trajectory
tokens and fed the CoC noise. Now each joint rollout gets TWO rewards and two
group-normalised advantages: an ADE reward on the trajectory span, the
GT-grounded text rules on the CoC span, and a self-consistency term (the CoC's
stated maneuver scored against the kinematics of the student's OWN decoded
plan) on both - the only term that builds the chain instead of assuming it.
The CoC is sampled at `coc_temperature` in rollouts so the sharp 4-camera
student's groups hold different reasonings to choose between.
"""
from __future__ import annotations
import contextlib
import functools
import json
import logging
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from . import checkpoint, losses
from .config import load_config
from .data.dataset import Stage1Dataset, collate_student, move_batch
from .data.splits import load_split
from .eval import coc_score, gt_reward
from .student import lora
from .student.edge_wrapper import EdgeStudent

log = logging.getLogger(__name__)

STOPS = ("<|cot_end|>", "</think>")


# ------------------------------------------------------------ pure parts ----

def decode_coc(tokenizer, ids: list[int], vocab_lo: int) -> tuple[str, bool]:
    """Student ids -> (text, terminated). Same contract as 05a/05b `_decode`."""
    text = tokenizer.decode([i for i in ids if i < vocab_lo], skip_special_tokens=True)
    for s in STOPS:
        if s in text:
            return text.split(s, 1)[0].strip(), True
    return text.strip(), False


def coc_reward(student_text: str, teacher_text: str, terminated: bool,
               n_tokens: int, w) -> tuple[float, dict]:
    """Scalar reward for one rollout, plus the fields it came from.

    `w` carries the weights (cfg.stage1_rl.reward). Not terminating, or running
    past `max_tokens`, is a hard failure regardless of content: the CoC's KV is
    what stage 2 attends, and an unterminated one is unusable.
    """
    if not terminated or n_tokens > int(w.max_tokens):
        return float(w.fail), {"fail": True}
    s = coc_score.score_pair(student_text, teacher_text)
    if s is None:                       # empty teacher trace: nothing to score against
        return 0.0, {"unscored": True}
    r = 0.0
    if s["maneuver_match"] is not None:
        r += float(w.maneuver) * (1.0 if s["maneuver_match"] else 0.0)
    if s["direction_match"] is not None:
        r += float(w.direction) * (1.0 if s["direction_match"] else 0.0)
    if s["obj_recall"] is not None:
        r += float(w.objects) * float(s["obj_recall"])
    if s["false_clear"]:
        r -= float(w.false_clear)
    return r, s


def group_advantages(rewards: list[float], eps: float = 1e-4) -> list[float] | None:
    """Group-normalised advantages, or None when the group has no signal."""
    n = len(rewards)
    mu = sum(rewards) / n
    var = sum((r - mu) ** 2 for r in rewards) / n
    if var < 1e-8:
        return None
    sd = math.sqrt(var)
    return [(r - mu) / (sd + eps) for r in rewards]


@contextlib.contextmanager
def adapters_off(root: torch.nn.Module):
    """Reference policy = the same weights with every LoRA branch scaled to 0.

    `LoraInjectedLinear.forward` is `base(x) + (alpha / r) * B(A(x))`
    (cosmos_framework/utils/generator/lora.py), so zeroing `_lora_alpha` for the
    duration of a forward gives the merged SFT checkpoint's logits exactly.
    """
    mods = [m for _, m in lora.lora_modules(root)]     # yields (path, module)
    saved = [m._lora_alpha for m in mods]
    for m in mods:
        m._lora_alpha = 0.0
    try:
        yield
    finally:
        for m, a in zip(mods, saved):
            m._lora_alpha = a


# ------------------------------------------------------------ the loop ----

def _coc_logp(student, batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probs of the CoC positions, (B, n) with a validity mask."""
    return _span_logps(student, batch)["coc"]


def _span_logps(student, batch) -> dict:
    """One forward; per-token log-probs for the CoC span and, when the batch
    carries trajectory positions (joint rollouts, D-039), the trajectory span."""
    out = student.ar_forward(batch, capture_layers=())
    res = {}
    for key, pos in (("coc", "coc_pos"), ("traj", "traj_pos")):
        n = int(batch[pos].sum(1).max())
        if n == 0:
            continue
        lg, tgt, ok = losses.gather_targets(out["logits"], batch["input_ids"], batch[pos], n)
        res[key] = (-F.cross_entropy(lg.float().transpose(1, 2), tgt, reduction="none"), ok)
    return res


def _rollout_traj(student, item, gen_ctx, G, pad_id, max_new, coc_temperature=None):
    """Joint CoC + trajectory rollouts for one prompt; returns per-sample
    (text, terminated, n_coc_tokens, bins|None, ade|None, plan_xyz|None).
    `plan_xyz` is the decoded (H, 3) ego-frame plan the ADE was computed on;
    the per-span reward scores the CoC against it (self-consistency)."""
    student.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        gb = move_batch(collate_student([{"student": gen_ctx}] * G, pad_id))
        outs = student.generate_coc_and_traj(gb, max_coc_tokens=max_new,
                                             coc_temperature=coc_temperature)
    student.train()
    vocab_lo = student.new_token_range[0]
    valid = [j for j, o in enumerate(outs) if o["bins"] is not None]
    ades, plans = {}, {}
    if valid:
        hx = torch.as_tensor(item["hist_xyz"]).float()[None]
        hr = torch.as_tensor(item["hist_rot"]).float()[None]
        toks = torch.stack([outs[j]["bins"] for j in valid])
        xyz = student.detokenize_traj(toks, hx.expand(len(valid), -1, -1),
                                      hr.expand(len(valid), -1, -1, -1))
        for jj, j in enumerate(valid):
            plans[j] = xyz[jj].float().cpu().numpy()
            ades[j] = gt_reward.ade_xy(plans[j], item["gt_future_xyz"])
    rows = []
    for j, o in enumerate(outs):
        text, term = decode_coc(student.tokenizer, o["coc_ids"], vocab_lo)
        rows.append((text, term, len(o["coc_ids"]), o["bins"], ades.get(j), plans.get(j)))
    return rows


def _gen_ctx(ds, i, item, route: bool):
    """Generation prompt for window i, with the GT-derived route hint when
    `route` is on. The dataset's own context has no route; rebuild it."""
    if not route:
        return item["student"], None
    path = ds.shards[i]
    window = ds._window(path.parent.name, int(path.stem))
    hint = gt_reward.route_hint(gt_reward.kinematics(item["gt_future_xyz"]))
    return ds.ctx.build(window, coc_text=None, nav_text=hint), hint


def _rate(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _val_eval(cfg, student, ds_val, n: int, pad_id: int, max_new: int, route: bool,
              traj: bool = False) -> dict:
    """Free-running check on val. Teacher-relative fields (maneuver acc vs the
    cached trace) AND teacher-free ones (`gt_reward.gt_metrics`): consistency
    with the driver's speed/path, GT false-clear, ungrounded hazard mentions,
    direction vs the direction taken. `gt_score` drives selection."""
    student.eval()
    rows = []
    vocab_lo = student.new_token_range[0]
    with torch.no_grad():
        for i in range(min(n, len(ds_val))):
            item = ds_val[i]
            ctx, _ = _gen_ctx(ds_val, i, item, route)
            ade = None
            if traj:
                text, term, _, _, ade, _ = _rollout_traj(student, item, ctx, 1, pad_id, max_new)[0]
            else:
                batch = move_batch(collate_student([{"student": ctx}], pad_id))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ids = student.generate_coc_text(batch, max_new_tokens=max_new)[0]
                text, term = decode_coc(student.tokenizer, ids, vocab_lo)
            k = gt_reward.kinematics(item["gt_future_xyz"])
            g = gt_reward.gt_metrics(text, k)
            s = coc_score.score_pair(text, str(item["coc_text"])) or {}
            rows.append({**g, "terminated": term, "ade": ade,
                         "maneuver_match": s.get("maneuver_match"),
                         "obj_precision": s.get("obj_precision")})
    student.train()
    consistent = _rate([r["consistent"] for r in rows]) or 0.0
    gt_fc = _rate([r["gt_false_clear"] for r in rows]) or 0.0
    ade_mean = _rate([r["ade"] for r in rows])
    return {
        "n": len(rows),
        "traj_ade": ade_mean,
        "neg_ade": (-ade_mean) if ade_mean is not None else None,
        "traj_decoded": _rate([r["ade"] is not None for r in rows]),
        "gt_score": consistent - gt_fc,
        "gt_consistent": consistent,
        "gt_false_clear": gt_fc,
        "hazard_ungrounded": _rate([r["hazard_ungrounded"] for r in rows]),
        "direction_ok": _rate([r["direction_ok"] for r in rows]),
        "maneuver_acc": _rate([r["maneuver_match"] for r in rows]),
        "obj_precision": _rate([r["obj_precision"] for r in rows]),
        "termination": _rate([r["terminated"] for r in rows]),
        "mix": dict(Counter(r["student_maneuver"] for r in rows).most_common()),
    }


def main(cfg_path: str, smoke: bool = False, run_name: str | None = None) -> None:
    cfg = load_config(cfg_path)
    rl = cfg.stage1_rl
    if run_name:            # a smoke gets its own dir, so its rows never land in the real curve
        cfg.raw["stage1_rl"]["run_name"] = run_name
    G = int(rl.group_size)
    B = int(rl.prompts_per_step)
    steps = int(rl.steps)
    eval_n = int(rl.eval_windows)
    if smoke:
        G, B, steps, eval_n = 4, 2, 2, 8
        log.info("SMOKE: G=%d prompts=%d steps=%d eval=%d", G, B, steps, eval_n)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(int(rl.get("seed", 0)))
    # RL-only camera set (D-038): the reward is valid against what a driver can
    # see, and the 1-camera teacher reproduced its own committed labels 48% of
    # the time vs 72% on 4. The context builder reads data.cameras at call time.
    if rl.get("cameras"):
        cfg.raw["data"]["cameras"] = list(rl.cameras)
    log.info("cameras: %s", list(cfg.data.raw["cameras"]))
    route = bool(rl.get("route_hint", False))
    mode = str(rl.reward.get("mode", "teacher"))
    coc_temp = rl.get("coc_temperature")           # rollout-only; val decodes at `temperature`
    coc_temp = float(coc_temp) if coc_temp is not None else None
    log.info("reward mode: %s   route hint: %s   coc_temperature: %s", mode, route, coc_temp)

    # --- policy: merged SFT checkpoint + fresh adapters ------------------
    student = EdgeStudent(cfg).cuda()
    student.extend_trajectory_vocab(torch.load(
        Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt", weights_only=False))
    student.lm._ensure_vision_tower()
    init = Path(rl.init_ckpt)
    meta = checkpoint.load_into(student, init)
    log.info("init policy <- %s (epoch=%s coc_nll=%s)", init, meta.get("epoch"), meta.get("val_coc_nll"))
    groups = student.param_groups_stage1()          # injects LoRA, freezes the rest
    lora_params = groups[0]["params"]
    log.info("LoRA: %s", student.lora_stats)
    for p in groups[1]["params"]:                   # new-token rows: SFT-trained, hold fixed
        p.requires_grad_(False)
    opt = torch.optim.AdamW(lora_params, lr=float(rl.lr), weight_decay=0.0, betas=(0.9, 0.99))
    student.train()
    pad_id = student.tokenizer.pad_token_id
    vocab_lo = student.new_token_range[0]

    # --- data: generation prompts; the scoring contexts are rebuilt per sample
    ds = Stage1Dataset(cfg, student.context_builder(),
                       clip_ids=load_split(cfg, "train"), cot_generation=True)
    ds_val = Stage1Dataset(cfg, student.context_builder(),
                           clip_ids=load_split(cfg, "val"), cot_generation=True)
    order = list(range(len(ds)))
    random.shuffle(order)
    log.info("train prompts: %d   val eval windows: %d", len(ds), eval_n)
    # Prompt selection. Job 315's first steps at uniform prompts and G=8 skipped
    # 69-81% of groups: on an easy FOLLOW scene all G rollouts agree and there is
    # no advantage. Drawing prompts by the teacher's maneuver class (the run-6
    # sampler weights) puts variance-rich scenes in front of the policy more
    # often. Unlike SFT this does not bias what is learned - advantages are
    # relative within a group - it only decides which groups exist at all.
    prompt_w = None
    p_alpha = float(rl.get("prompt_alpha", 0.0))
    if p_alpha > 0:
        from .data.sampling import log_mix, maneuver_weights
        prompt_w, mix = maneuver_weights(ds.shards, p_alpha)
        log_mix(mix, p_alpha)

    # decode at the setting the student is judged at
    cfg.raw["teacher"]["gen_temperature"] = float(rl.temperature)
    cfg.raw["teacher"]["gen_top_p"] = float(rl.top_p)
    max_new = int(rl.max_new_tokens)
    beta, ce_aux = float(rl.kl_beta), float(rl.ce_aux)

    run_dir = Path(cfg.paths.runs_root) / "stage1_rl" / (rl.get("run_name") or "run")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_acc, cursor = float("-inf"), 0   # any first eval is a best (neg_ade is < -1)
    hist = open(run_dir / "steps.jsonl", "a")

    perspan = (mode == "perspan")
    traj_mode = (mode == "traj") or perspan          # both roll out CoC + trajectory
    w_traj = float(rl.get("traj_loss_weight", 1.0))
    base = _val_eval(cfg, student, ds_val, eval_n, pad_id, max_new, route, traj_mode)
    log.info("step 0 val: %s", json.dumps(base))

    for step in range(1, steps + 1):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        stats = Counter()
        rew_all, rew_coc_all, kl_all, n_tok = [], [], [], 0
        mix = Counter()
        for _ in range(B):
            if prompt_w is not None:
                i = int(torch.multinomial(prompt_w, 1).item())
            else:
                if cursor >= len(order):
                    random.shuffle(order); cursor = 0
                i = order[cursor]; cursor += 1
            item = ds[i]
            path = ds.shards[i]
            window = ds._window(path.parent.name, int(path.stem))
            teacher = str(item["coc_text"]).strip()
            k = gt_reward.kinematics(item["gt_future_xyz"])
            hint = gt_reward.route_hint(k) if route else None
            gen_ctx = ds.ctx.build(window, coc_text=None, nav_text=hint) if route else item["student"]

            # ---- rollouts: G samples of the same prompt ----
            texts, rewards, terms, bins_list = [], [], [], []
            rew_coc = []                                   # perspan: the CoC span's own reward
            if traj_mode:
                samples = _rollout_traj(student, item, gen_ctx, G, pad_id, max_new, coc_temp)
            else:
                student.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    gb = move_batch(collate_student([{"student": gen_ctx}] * G, pad_id))
                    rows = student.generate_coc_text(gb, max_new_tokens=max_new)
                student.train()
                samples = [(*decode_coc(student.tokenizer, ids, vocab_lo), len(ids), None, None, None)
                           for ids in rows]
            for text, term, n_ids, bins, ade, plan in samples:
                if perspan:
                    r, r_c, s = gt_reward.perspan_reward(ade, plan, text, term, n_ids, k, teacher,
                                                         rl.reward)
                    rew_coc.append(r_c)
                    if ade is not None:
                        stats["ade_sum"] += ade; stats["ade_n"] += 1
                elif traj_mode:
                    r, s = gt_reward.traj_reward(ade, text, term, n_ids, k, teacher, rl.reward)
                    if ade is not None:
                        stats["ade_sum"] += ade; stats["ade_n"] += 1
                elif mode == "gt":
                    r, s = gt_reward.gt_reward(text, term, n_ids, k, teacher, rl.reward)
                else:
                    r, s = coc_reward(text, teacher, term, n_ids, rl.reward)
                bins_list.append(bins)
                texts.append(text); rewards.append(r); terms.append(term)
                mix[s.get("student_maneuver") if isinstance(s, dict) else None] += 1
                stats["fail"] += int(bool(s.get("fail"))) if isinstance(s, dict) else 0
                stats["false_clear"] += int(bool(s.get("false_clear"))) if isinstance(s, dict) else 0
                stats["man_match"] += int(bool(s.get("maneuver_match"))) if isinstance(s, dict) else 0
                for comp in ("kin", "dir", "hazard", "teacher", "self"):  # gt / perspan components
                    if isinstance(s, dict) and comp in s:
                        stats[comp] += float(s[comp])
            rew_all += rewards
            rew_coc_all += rew_coc
            stats["groups"] += 1
            # Advantages. One list per span: in perspan mode the trajectory
            # span's comes from r_traj and the CoC span's from r_coc, each
            # group-normalised on its own with its own zero-variance skip (a
            # flat span gets zero advantage; the group is dropped only when
            # both are flat). Every other mode applies the one advantage to
            # whatever spans the rollout has.
            adv = group_advantages(rewards)
            adv_c = group_advantages(rew_coc) if perspan else adv
            if perspan:
                stats["flat_traj"] += int(adv is None)
                stats["flat_coc"] += int(adv_c is None)
            if adv is None and adv_c is None:
                stats["skipped"] += 1
                if not smoke:
                    continue
                # Smoke must exercise the backward path even when every group is
                # flat (easy prompts at G=4 usually are): zero advantage leaves
                # only the KL + CE terms, which is enough to run it end to end.
                adv = adv_c = [0.0] * len(rewards)
            adv = adv if adv is not None else [0.0] * len(rewards)
            adv_c = adv_c if adv_c is not None else [0.0] * len(rewards)

            # ---- policy + reference log-probs on the sampled CoCs ----
            keep = [(t, a, ac, bn) for t, a, ac, bn in zip(texts, adv, adv_c, bins_list) if t]
            if not keep:                                    # empty text: no tokens
                stats["skipped"] += 1
                continue
            # Scoring layout = the training layout: CoC teacher-forced from the
            # sample; in traj mode the sampled bins follow it (a failed sample
            # has no bins and is scored on its CoC alone).
            ctxs = [{"student": (ds.ctx.build(window, coc_text=t, traj_bins=bn.tolist(), nav_text=hint)
                                 if bn is not None else
                                 ds.ctx.build(window, coc_text=t, for_generation=True, nav_text=hint))}
                    for t, _, _, bn in keep]
            advs = {"traj": [a for _, a, _, _ in keep], "coc": [ac for _, _, ac, _ in keep]}
            # Score in chunks with gradient accumulation: the full-vocab logits
            # over every position of G ~1.7k-token rows do not fit next to the
            # activations for backward at G=16 (or at G=4 on the Ada's cap).
            # Each chunk's per-span mean is weighted by its share of the rows,
            # so the sum equals the un-chunked per-span mean.
            chunk = int(rl.get("score_chunk", 8))
            for c0 in range(0, len(ctxs), chunk):
                cctx = ctxs[c0:c0 + chunk]
                share = len(cctx) / len(ctxs)
                sb = move_batch(collate_student(cctx, pad_id))
                A_span = {sp: torch.tensor(advs[sp][c0:c0 + chunk], device="cuda").view(-1, 1)
                          for sp in advs}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    with torch.no_grad(), adapters_off(student._decoder_layers()):
                        ref = _span_logps(student, sb)
                    pol = _span_logps(student, sb)
                pg = torch.zeros((), device="cuda")
                for span, wspan in (("coc", 1.0), ("traj", w_traj)):
                    if span not in pol:
                        continue
                    logp, ok = pol[span]; ref_logp, _ = ref[span]
                    A = A_span[span]
                    okf = ok.float()
                    ratio = torch.exp(logp - logp.detach())           # 1 with grad d logp
                    d = ref_logp - logp
                    kl = torch.exp(d) - d - 1.0                       # k3, >= 0
                    per_tok = -A * ratio + beta * kl
                    # Per-SPAN mean: 128 trajectory tokens must not drown 14 CoC tokens.
                    pg = pg + wspan * share * (per_tok * okf).sum() / okf.sum().clamp_min(1) / B
                    kl_all.append(float((kl * okf).sum() / okf.sum().clamp_min(1)))
                    n_tok += int(okf.sum())
                pg.backward()
                del pol, ref, sb

            # ---- CE anchor on the teacher's trace ----
            if ce_aux > 0 and teacher:
                tb = move_batch(collate_student(
                    [{"student": ds.ctx.build(window, coc_text=teacher, for_generation=True,
                                              nav_text=hint)}], pad_id))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tl, tok_ = _coc_logp(student, tb)
                ce = -(tl * tok_.float()).sum() / tok_.float().sum().clamp_min(1)
                (ce_aux * ce / B).backward()
                stats["ce"] += float(ce)

        gn = torch.nn.utils.clip_grad_norm_(lora_params, float(rl.grad_clip))
        opt.step()
        rec = {"step": step, "reward": sum(rew_all) / max(len(rew_all), 1),
               "man_match": stats["man_match"] / max(len(rew_all), 1),
               "false_clear": stats["false_clear"] / max(len(rew_all), 1),
               "fail": stats["fail"] / max(len(rew_all), 1),
               "skipped_groups": stats["skipped"] / max(stats["groups"], 1),
               "kl": sum(kl_all) / max(len(kl_all), 1), "ce": stats["ce"] / B,
               "grad_norm": float(gn), "tokens": n_tok, "sec": round(time.time() - t0, 1),
               "mix": dict(mix.most_common(6))}
        if perspan:
            N = max(len(rew_all), 1)
            rec.update(reward_coc=sum(rew_coc_all) / N, ade=stats["ade_sum"] / max(stats["ade_n"], 1),
                       decoded=stats["ade_n"] / N, kin=stats["kin"] / N, dir=stats["dir"] / N,
                       hazard=stats["hazard"] / N, self_consistency=stats["self"] / N,
                       flat_traj=stats["flat_traj"] / max(stats["groups"], 1),
                       flat_coc=stats["flat_coc"] / max(stats["groups"], 1))
            log.info("step %d/%d r_traj %.3f r_coc %.3f [ade %.2f m decoded %.2f kin %+.2f dir %+.2f "
                     "haz %+.2f self %+.2f] fail %.2f flat traj/coc %.2f/%.2f kl %.4f ce %.3f |g| %.2f  "
                     "%.0fs  %s", step, steps, rec["reward"], rec["reward_coc"], rec["ade"],
                     rec["decoded"], rec["kin"], rec["dir"], rec["hazard"], rec["self_consistency"],
                     rec["fail"], rec["flat_traj"], rec["flat_coc"], rec["kl"], rec["ce"],
                     rec["grad_norm"], rec["sec"], rec["mix"])
        elif traj_mode:
            N = max(len(rew_all), 1)
            rec.update(ade=stats["ade_sum"] / max(stats["ade_n"], 1), decoded=stats["ade_n"] / N,
                       kin=stats["kin"] / N, dir=stats["dir"] / N)
            log.info("step %d/%d reward %.3f [ade %.2f m decoded %.2f kin %+.2f dir %+.2f] fail %.2f skip %.2f "
                     "kl %.4f ce %.3f |g| %.2f  %.0fs  %s", step, steps, rec["reward"], rec["ade"],
                     rec["decoded"], rec["kin"], rec["dir"], rec["fail"], rec["skipped_groups"],
                     rec["kl"], rec["ce"], rec["grad_norm"], rec["sec"], rec["mix"])
        elif mode == "gt":
            N = max(len(rew_all), 1)
            rec.update(kin=stats["kin"] / N, dir=stats["dir"] / N, hazard=stats["hazard"] / N,
                       teacher=stats["teacher"] / N)
            log.info("step %d/%d reward %.3f [kin %+.2f dir %+.2f haz %+.2f tch %.2f] fail %.2f skip %.2f "
                     "kl %.4f ce %.3f |g| %.2f  %.0fs  %s", step, steps, rec["reward"], rec["kin"],
                     rec["dir"], rec["hazard"], rec["teacher"], rec["fail"], rec["skipped_groups"],
                     rec["kl"], rec["ce"], rec["grad_norm"], rec["sec"], rec["mix"])
        else:
            log.info("step %d/%d reward %.3f man %.2f fc %.2f fail %.2f skip %.2f kl %.4f ce %.3f |g| %.2f  %.0fs  %s",
                     step, steps, rec["reward"], rec["man_match"], rec["false_clear"], rec["fail"],
                     rec["skipped_groups"], rec["kl"], rec["ce"], rec["grad_norm"], rec["sec"], rec["mix"])
        hist.write(json.dumps(rec) + "\n"); hist.flush()

        if step % int(rl.eval_every) == 0 or step == steps:
            v = _val_eval(cfg, student, ds_val, eval_n, pad_id, max_new, route, traj_mode)
            log.info("step %d val: %s", step, json.dumps(v))
            hist.write(json.dumps({"step": step, "val": v}) + "\n"); hist.flush()
            sel_key = str(rl.get("select_on", "gt_score"))
            sel = float(v[sel_key] if v.get(sel_key) is not None else -1e9)
            if not smoke:
                # Every eval keeps its adapters (~75 MB): nothing is lost to a
                # noisy selector, and any step can be scored on full n later.
                checkpoint.save_adapters(student, run_dir / f"step-{step:04d}", stage="stage1_rl",
                                         step=step, init_ckpt=str(init), val=v)
            if not smoke and sel > best_acc:
                best_acc = sel
                checkpoint.save(student, run_dir / "best", stage="stage1_rl", step=step,
                                select_on=sel_key, val_select=float(best_acc),
                                val_maneuver_acc=float(v["maneuver_acc"] or 0.0), init_ckpt=str(init))
                log.info("  <- new best (val %s %.3f), saved", sel_key, best_acc)
    if smoke:
        log.info("SMOKE OK  peak GPU %.1f GiB", torch.cuda.max_memory_allocated() / 2**30)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run-name", default=None,
                    help="override stage1_rl.run_name (give a smoke its own steps.jsonl)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(a.config, smoke=a.smoke, run_name=a.run_name)
