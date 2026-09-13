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
from .eval import coc_score
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
    out = student.ar_forward(batch, capture_layers=())
    n = int(batch["coc_pos"].sum(1).max())
    lg, tgt, ok = losses.gather_targets(out["logits"], batch["input_ids"], batch["coc_pos"], n)
    logp = -F.cross_entropy(lg.float().transpose(1, 2), tgt, reduction="none")
    return logp, ok


def _val_eval(cfg, student, ds_val, n: int, pad_id: int, max_new: int) -> dict:
    """Light free-running check on val: maneuver accuracy, false-clear, mix."""
    student.eval()
    rows = []
    vocab_lo = student.new_token_range[0]
    with torch.no_grad():
        for i in range(min(n, len(ds_val))):
            item = ds_val[i]
            batch = move_batch(collate_student([item], pad_id))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ids = student.generate_coc_text(batch, max_new_tokens=max_new)[0]
            text, term = decode_coc(student.tokenizer, ids, vocab_lo)
            s = coc_score.score_pair(text, str(item["coc_text"]))
            if s is not None:
                s["terminated"] = term
                rows.append(s)
    student.train()
    haz = [r for r in rows if r["hazard_window"]]
    mm = [r["maneuver_match"] for r in rows if r["maneuver_match"] is not None]
    return {
        "n": len(rows),
        "maneuver_acc": sum(mm) / max(len(mm), 1),
        "false_clear": sum(r["false_clear"] for r in haz) / max(len(haz), 1),
        "termination": sum(r["terminated"] for r in rows) / max(len(rows), 1),
        "obj_precision": sum(r["obj_precision"] for r in rows if r["obj_precision"] is not None)
                         / max(sum(1 for r in rows if r["obj_precision"] is not None), 1),
        "mix": dict(Counter(r["student_maneuver"] for r in rows).most_common()),
    }


def main(cfg_path: str, smoke: bool = False) -> None:
    cfg = load_config(cfg_path)
    rl = cfg.stage1_rl
    G = int(rl.group_size)
    B = int(rl.prompts_per_step)
    steps = int(rl.steps)
    eval_n = int(rl.eval_windows)
    if smoke:
        G, B, steps, eval_n = 4, 2, 2, 8
        log.info("SMOKE: G=%d prompts=%d steps=%d eval=%d", G, B, steps, eval_n)
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(int(rl.get("seed", 0)))

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
    best_acc, cursor = -1.0, 0
    hist = open(run_dir / "steps.jsonl", "a")

    base = _val_eval(cfg, student, ds_val, eval_n, pad_id, max_new)
    log.info("step 0 val: %s", json.dumps(base))

    for step in range(1, steps + 1):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        stats = Counter()
        rew_all, kl_all, n_tok = [], [], 0
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

            # ---- rollouts: G samples of the same prompt ----
            student.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                gb = move_batch(collate_student([item] * G, pad_id))
                rows = student.generate_coc_text(gb, max_new_tokens=max_new)
            student.train()
            texts, rewards, terms = [], [], []
            for ids in rows:
                text, term = decode_coc(student.tokenizer, ids, vocab_lo)
                r, s = coc_reward(text, teacher, term, len(ids), rl.reward)
                texts.append(text); rewards.append(r); terms.append(term)
                mix[s.get("student_maneuver") if isinstance(s, dict) else None] += 1
                stats["fail"] += int(bool(s.get("fail"))) if isinstance(s, dict) else 0
                stats["false_clear"] += int(bool(s.get("false_clear"))) if isinstance(s, dict) else 0
                stats["man_match"] += int(bool(s.get("maneuver_match"))) if isinstance(s, dict) else 0
            rew_all += rewards
            adv = group_advantages(rewards)
            stats["groups"] += 1
            if adv is None:
                stats["skipped"] += 1
                if not smoke:
                    continue
                # Smoke must exercise the backward path even when every group is
                # flat (easy prompts at G=4 usually are): zero advantage leaves
                # only the KL + CE terms, which is enough to run it end to end.
                adv = [0.0] * len(rewards)

            # ---- policy + reference log-probs on the sampled CoCs ----
            keep = [(t, a) for t, a in zip(texts, adv) if t]      # empty text: no tokens
            if not keep:
                stats["skipped"] += 1
                continue
            ctxs = [{"student": ds.ctx.build(window, coc_text=t, for_generation=True)}
                    for t, _ in keep]
            sb = move_batch(collate_student(ctxs, pad_id))
            A = torch.tensor([a for _, a in keep], device="cuda").view(-1, 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad(), adapters_off(student._decoder_layers()):
                    ref_logp, _ = _coc_logp(student, sb)
                logp, ok = _coc_logp(student, sb)
            okf = ok.float()
            ratio = torch.exp(logp - logp.detach())               # 1 with grad d logp
            d = ref_logp - logp
            kl = torch.exp(d) - d - 1.0                           # k3, >= 0
            per_tok = -A * ratio + beta * kl
            pg = (per_tok * okf).sum() / okf.sum().clamp_min(1) / B
            pg.backward()
            kl_all.append(float((kl * okf).sum() / okf.sum().clamp_min(1)))
            n_tok += int(okf.sum())

            # ---- CE anchor on the teacher's trace ----
            if ce_aux > 0 and teacher:
                tb = move_batch(collate_student(
                    [{"student": ds.ctx.build(window, coc_text=teacher, for_generation=True)}], pad_id))
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
        log.info("step %d/%d reward %.3f man %.2f fc %.2f fail %.2f skip %.2f kl %.4f ce %.3f |g| %.2f  %.0fs  %s",
                 step, steps, rec["reward"], rec["man_match"], rec["false_clear"], rec["fail"],
                 rec["skipped_groups"], rec["kl"], rec["ce"], rec["grad_norm"], rec["sec"], rec["mix"])
        hist.write(json.dumps(rec) + "\n"); hist.flush()

        if step % int(rl.eval_every) == 0 or step == steps:
            v = _val_eval(cfg, student, ds_val, eval_n, pad_id, max_new)
            log.info("step %d val: %s", step, json.dumps(v))
            hist.write(json.dumps({"step": step, "val": v}) + "\n"); hist.flush()
            if not smoke and v["maneuver_acc"] > best_acc:
                best_acc = v["maneuver_acc"]
                checkpoint.save(student, run_dir / "best", stage="stage1_rl", step=step,
                                val_maneuver_acc=float(best_acc), init_ckpt=str(init))
                log.info("  <- new best (val maneuver acc %.3f), saved", best_acc)
    if smoke:
        log.info("SMOKE OK  peak GPU %.1f GiB", torch.cuda.max_memory_allocated() / 2**30)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(a.config, smoke=a.smoke)
