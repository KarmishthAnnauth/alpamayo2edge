"""Phase 2, step 2: the CoC on CARLA. Generate it once per Bench2Drive window
with the phase-2 init (RL run 7 step 25 folded into run-351/best), cache it for
the flow head to condition on, and score it against the EXPERT's kinematics
with the same driver-grounded rows 05f uses on PhysicalAI-AV. That table is the
CARLA-transfer check the plan called for: the CoC was distilled on real footage
and has never seen a render.

    bash scripts/ada_run.sh scripts/05j_b2d_coc.py generate            # all windows, resumable
    bash scripts/ada_run.sh scripts/05j_b2d_coc.py generate --n 200    # a look
    python scripts/05j_b2d_coc.py score [--split val]                  # CPU

Output: `<b2d_cache>/coc_rl7s25.jsonl` (config `stage2.coc_cache`), one row per
window {clip, window, student, terminated}; `score` reads it back.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

from distill.config import load_config                              # noqa: E402
from distill.data import bench2drive as b2d                         # noqa: E402
from distill.data.dataset import move_batch                         # noqa: E402
from distill.eval import gt_reward                                  # noqa: E402

log = logging.getLogger("b2d_coc")

# PhysicalAI-AV full-val reference rows (D-058, two draws): what the same
# checkpoint does on real footage. Printed next to the CARLA numbers.
REF = {"braked_slow": (0.385, 0.216), "stopped": (0.750, 0.690), "hard": (0.379, 0.165),
       "false_clear": (0.288, 0.190), "consistent": (0.780, 0.691), "nudge": (0.035, 0.099)}


def _decode(student, ids: list[int]) -> tuple[str, bool]:
    lo = student.new_token_range[0]
    text = student.tokenizer.decode([i for i in ids if i < lo], skip_special_tokens=True)
    for s in ("<|cot_end|>", "</think>"):
        if s in text:
            return text.split(s)[0].strip(), True
    return text.strip(), False


def generate(a, cfg) -> int:
    from distill import train_stage2 as ts2
    out_path = Path(a.out or cfg.stage2.coc_cache)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["clip"], int(r["window"])))
        log.info("resuming: %d windows already in %s", len(done), out_path)

    if a.ckpt:
        # a merged stage-2 checkpoint (AR LoRA folded): what the CoC looks like
        # AFTER the flow loss touched the reasoner (run 3 drift check)
        from distill.student.edge_wrapper import EdgeStudent
        from distill import checkpoint
        student = EdgeStudent(cfg).cuda()
        student.extend_trajectory_vocab(torch.load(
            Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt", weights_only=False))
        student.lm._ensure_vision_tower()
        meta = checkpoint.load_into(student, Path(a.ckpt))
        log.info("stage-2 checkpoint %s (epoch=%s)", a.ckpt, meta.get("epoch"))
    else:
        student = ts2.load_student(cfg)              # init ckpt + RL adapters folded
    student.eval()
    root = Path(cfg.paths.b2d_cache_root)
    clip_ids = b2d.load_b2d_split(root, a.split) if a.split else None
    ds = b2d.Bench2DriveDataset(root, student.context_builder(), clip_ids=clip_ids,
                                coc_lookup=None, for_generation=True)   # free-running prompt
    pad_id = student.tokenizer.pad_token_id
    todo = [i for i in range(len(ds))
            if (ds.shards[i].parent.name, int(ds.shards[i].stem)) not in done]
    if a.n is not None:
        todo = todo[:a.n]
    log.info("%d windows total, %d to generate", len(ds), len(todo))
    t0 = time.time()
    term = 0
    with open(out_path, "a") as f:
        for j, i in enumerate(todo):
            item = ds[i]
            batch = move_batch(b2d.collate_b2d([item], pad_id))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = student.generate_coc_text(batch, max_new_tokens=a.max_new_tokens)
            text, terminated = _decode(student, out[0])
            term += terminated
            f.write(json.dumps({"clip": item["clip_id"], "window": int(item["window_idx"]),
                                "student": text, "terminated": terminated,
                                "route_hint": item["route_hint"]}) + "\n")
            if (j + 1) % 50 == 0:
                f.flush()
                el = time.time() - t0
                log.info("  %d/%d  %.2f s/window  terminated %.3f  eta %.0f min", j + 1, len(todo),
                         el / (j + 1), term / (j + 1), el / (j + 1) * (len(todo) - j - 1) / 60)
    log.info("done: %d generated in %.0f s -> %s", len(todo), time.time() - t0, out_path)
    return 0


def score(a, cfg) -> int:
    from distill.eval.coc_score import parse
    SLOWING = ("STOP", "SLOW", "YIELD")          # 05f's definition
    root = Path(cfg.paths.b2d_cache_root)
    path = Path(a.out or cfg.stage2.coc_cache)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    keep = None
    if a.split:
        keep = set(b2d.load_b2d_split(root, a.split))
        rows = [r for r in rows if r["clip"] in keep]
    if not rows:
        raise SystemExit("no rows to score")

    agg, n_braked, sev_n, sev_slow = [], 0, Counter(), Counter()
    braked_man, all_man = Counter(), Counter()
    flagged = {"n": 0, "slow": 0}          # expert's own should_brake flag over the horizon
    by_scn = {}
    for r in rows:
        with np.load(root / r["clip"] / f"{int(r['window']):02d}.npz", allow_pickle=True) as z:
            k = gt_reward.kinematics(z["gt_future_xyz"])
            sbf = float(z["should_brake_frac"]) if "should_brake_frac" in z else float("nan")
            scn = str(z["scenario"]) if "scenario" in z else r["clip"].split("_")[0]
        m = gt_reward.gt_metrics(r["student"], k)
        agg.append(m)
        man = parse(r["student"]).get("maneuver") or "NONE"
        all_man[man] += 1
        s = by_scn.setdefault(scn, {"n": 0, "braked": 0, "slow": 0, "fc": 0, "fc_n": 0})
        s["n"] += 1
        if k["braked_from_speed"]:
            n_braked += 1
            sev = ("stop" if k["vmin"] < 0.5 else "hard" if k["dv"] < -2.5 else "mild")
            sev_n[sev] += 1
            braked_man[man] += 1
            sev_slow[sev] += man in SLOWING
            s["braked"] += 1; s["slow"] += man in SLOWING
            if m["gt_false_clear"] is not None:
                s["fc_n"] += 1; s["fc"] += bool(m["gt_false_clear"])
        if sbf == sbf and sbf > 0.5:
            flagged["n"] += 1; flagged["slow"] += man in SLOWING

    def rate(vals):
        v = [x for x in vals if x is not None]
        return (sum(bool(x) for x in v) / len(v) if v else float("nan")), len(v)

    print(f"{path.name}: {len(rows)} windows, split {a.split or 'all'}   "
          f"(reference: RL7 step 25 / teacher on PhysicalAI-AV val, D-058)")
    print(f"  {'row':<34} {'CARLA':>8}  n     {'PAI RL7':>8} {'PAI teacher':>12}")
    c, n = rate(m["consistent"] for m in agg)
    print(f"  {'GT-consistent (checkable)':<34} {c:8.3f}  {n:<5d} {REF['consistent'][0]:8.3f} {REF['consistent'][1]:12.3f}")
    fc, n = rate(m["gt_false_clear"] for m in agg)
    print(f"  {'GT false-clear (braked)':<34} {fc:8.3f}  {n:<5d} {REF['false_clear'][0]:8.3f} {REF['false_clear'][1]:12.3f}")
    d, n = rate(m["direction_ok"] for m in agg)
    print(f"  {'direction stated ok (turns)':<34} {d:8.3f}  {n:<5d}")
    h, n = rate(m["hazard_ungrounded"] for m in agg)
    print(f"  {'hazard named, no reaction':<34} {h:8.3f}  {n:<5d}")
    slow = sum(braked_man[m] for m in SLOWING)
    print(f"  {'braked -> says slow/stop':<34} {slow / max(n_braked, 1):8.3f}  {n_braked:<5d} "
          f"{REF['braked_slow'][0]:8.3f} {REF['braked_slow'][1]:12.3f}")
    for sev, label, key in (("stop", "  expert STOPPED", "stopped"), ("hard", "  expert braked HARD", "hard"),
                            ("mild", "  expert slowed mildly", None)):
        if sev_n[sev]:
            ref = f"{REF[key][0]:8.3f} {REF[key][1]:12.3f}" if key else ""
            print(f"  {label:<34} {sev_slow[sev] / sev_n[sev]:8.3f}  {sev_n[sev]:<5d} {ref}")
    nudge = braked_man["NUDGE"] / max(n_braked, 1)
    print(f"  {'NUDGE share on braked':<34} {nudge:8.3f}  {n_braked:<5d} {REF['nudge'][0]:8.3f} {REF['nudge'][1]:12.3f}")
    if flagged["n"]:
        print(f"  {'should_brake>0.5 -> says slow/stop':<34} {flagged['slow'] / flagged['n']:8.3f}  {flagged['n']:<5d}")
    print(f"  terminated {sum(bool(r.get('terminated')) for r in rows) / len(rows):.3f}   "
          f"maneuver mix: {dict(all_man.most_common(8))}")
    print("  per scenario (n, braked, braked->slow/stop, false-clear):")
    for scn, s in sorted(by_scn.items(), key=lambda kv: -kv[1]["n"])[:a.top]:
        print(f"     {scn:<40} {s['n']:4d} {s['braked']:4d}  "
              f"{s['slow'] / max(s['braked'], 1):6.3f}  {s['fc'] / max(s['fc_n'], 1):6.3f}")
    if a.json:
        Path(a.json).write_text(json.dumps({"n": len(rows), "consistent": c, "false_clear": fc,
                                            "braked_slow": slow / max(n_braked, 1), "n_braked": n_braked,
                                            "direction_ok": d, "by_scenario": by_scn}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["generate", "score"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None, help="jsonl path (default stage2.coc_cache)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--split", default=None, help="restrict to split_<name>.json (generate and score)")
    ap.add_argument("--ckpt", default=None, help="generate: a merged stage-2 checkpoint instead of the init")
    ap.add_argument("--top", type=int, default=45)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(a.config)
    return generate(a, cfg) if a.cmd == "generate" else score(a, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
