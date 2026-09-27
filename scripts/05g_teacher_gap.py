"""How close is a student's free-running CoC to the TEACHER's, window by window?

`05f_coc_gt_score.py` grades one dump's student and teacher traces against the
driver and prints the two columns side by side. That answers "what are the two
rates?" but not "is the difference real?", and a single 500-window draw carries
+/-0.05-0.1 on the braking rows (D-052 addendum 2). This script closes both gaps:

  * takes SEVERAL draws of the same checkpoint (the noise rule from the runbook)
    and reports each row as mean over draws with the per-draw spread;
  * pairs student against teacher ON THE SAME WINDOW, so the comparison is not
    two independent rates but a within-window difference: discordant counts, an
    exact McNemar per draw, and a bootstrap CI over windows (the unit that is
    resampled - draws of one window are repeated measures, not extra windows);
  * separates AGREEING WITH THE TEACHER from BEING RIGHT. A student that
    reproduces the teacher's misses scores the same as the teacher on every
    driver row while being no better at seeing the road, so the joint-failure
    table (both wrong / only student right / only teacher right) is reported
    next to the rates.

    python scripts/05g_teacher_gap.py --dumps runs/<a>.jsonl runs/<b>.jsonl --split val

The teacher column is identical across draws (cached labels), so all of its
uncertainty is sampling of windows; the student's also includes decode noise,
which is what averaging the draws removes.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np                                                  # noqa: E402
from scipy.stats import binomtest                                   # noqa: E402

from distill.config import load_config                              # noqa: E402
from distill.eval import gt_reward                                  # noqa: E402
from distill.eval.coc_score import parse                            # noqa: E402

SLOWING = ("STOP", "SLOW", "YIELD")
SPEED_CLAIM = SLOWING + ("ADAPT_SPEED",)
BOOT = 4000


def load_draw(path: Path) -> dict:
    """dump -> {(clip, window): row}. Windows whose teacher trace is empty are
    dropped, matching 05b's own n_scored."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not str(r["teacher"]).strip():
            continue
        out[(r["clip"], int(r["window"]))] = r
    return out


def kin_for(cache_root: Path, key: tuple[str, int]) -> dict:
    with np.load(cache_root / key[0] / f"{key[1]:02d}.npz") as z:
        return gt_reward.kinematics(z["gt_future_xyz"])


def severity(k: dict) -> str | None:
    if not k["braked_from_speed"]:
        return None
    return "stop" if k["vmin"] < 0.5 else "hard" if k["dv"] < -2.5 else "mild"


# --------------------------------------------------------------- stats ----

def boot_ci(s_by_win: list[list[float]], t_by_win: list[float],
            rng: np.random.Generator) -> tuple[float, float]:
    """95% CI on (student - teacher) rate, resampling WINDOWS with replacement.
    Each window carries its own per-draw student values, so decode noise rides
    along with the window instead of being counted as independent evidence."""
    n = len(t_by_win)
    if n == 0:
        return (float("nan"), float("nan"))
    s_mean = np.array([np.mean(v) for v in s_by_win])
    t_arr = np.array(t_by_win, dtype=float)
    idx = rng.integers(0, n, size=(BOOT, n))
    diffs = s_mean[idx].mean(axis=1) - t_arr[idx].mean(axis=1)
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def paired_row(label: str, per_draw: list[dict], teacher: dict,
               rng: np.random.Generator) -> dict:
    """per_draw[i][key] = 0/1 for the windows where the metric is defined in
    draw i; teacher[key] likewise. Only windows defined in EVERY draw and for
    the teacher are paired, so the columns are never computed on different sets."""
    keys = set(teacher)
    for d in per_draw:
        keys &= set(d)
    keys = sorted(keys)
    s_by_win = [[d[k] for d in per_draw] for k in keys]
    t_by_win = [teacher[k] for k in keys]
    s_draw_rates = [float(np.mean([d[k] for k in keys])) if keys else float("nan")
                    for d in per_draw]
    s_rate = float(np.mean(s_draw_rates)) if keys else float("nan")
    t_rate = float(np.mean(t_by_win)) if keys else float("nan")
    lo, hi = boot_ci(s_by_win, t_by_win, rng)
    # Exact McNemar per draw: among windows where exactly one column is right,
    # how often is it the student? p = 1.0 means the two are indistinguishable.
    mc = []
    for d in per_draw:
        s_only = sum(1 for k in keys if d[k] > teacher[k])
        t_only = sum(1 for k in keys if d[k] < teacher[k])
        p = binomtest(s_only, s_only + t_only, 0.5).pvalue if (s_only + t_only) else 1.0
        mc.append((s_only, t_only, p))
    return {"label": label, "n": len(keys), "student": s_rate, "teacher": t_rate,
            "spread": (min(s_draw_rates), max(s_draw_rates)) if keys else (0, 0),
            "ci": (lo, hi), "mcnemar": mc, "keys": keys,
            "s_by_win": s_by_win, "t_by_win": t_by_win}


def fmt(r: dict, higher_is_better: bool = True) -> str:
    s, t = r["student"], r["teacher"]
    lo, hi = r["ci"]
    sp = r["spread"]
    verdict = "same" if lo <= 0 <= hi else ("BETTER" if (hi < 0) ^ higher_is_better else "WORSE")
    mcp = "/".join(f"{p:.2f}" for _, _, p in r["mcnemar"])
    return (f"  {r['label']:<32} {s:6.3f} [{sp[0]:.3f}-{sp[1]:.3f}]  {t:6.3f}  "
            f"{s - t:+6.3f} [{lo:+.3f},{hi:+.3f}]  n={r['n']:<4d} p={mcp:<14s} {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dumps", nargs="+", required=True,
                    help="one or more 05b dumps of the SAME checkpoint (different draws)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--label", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples", type=int, default=0,
                    help="print this many windows where the student is right and "
                         "the teacher is not, and vice versa, on the braking rows")
    a = ap.parse_args()
    cfg = load_config(a.config)
    cache_root = Path(cfg.paths.cache_root)
    rng = np.random.default_rng(a.seed)

    draws = [load_draw(Path(p)) for p in a.dumps]
    common = set(draws[0])
    for d in draws[1:]:
        common &= set(d)
    common = sorted(common)
    label = a.label or Path(a.dumps[0]).stem
    print(f"\n{'=' * 104}\n{label}: {len(draws)} draw(s), {len(common)} windows scored in "
          f"all of them, split {a.split}")
    for p, d in zip(a.dumps, draws):
        print(f"   {Path(p).name:<44} {len(d)} scored windows")

    kin = {k: kin_for(cache_root, k) for k in common}

    # -------------------------------------------------- driver-grounded ----
    # gt_metrics returns None where a row is not checkable on that window; a row
    # is paired only where BOTH columns are checkable (so a student that says
    # nothing checkable is never scored against a teacher that did).
    rows_s = [{k: gt_reward.gt_metrics(d[k]["student"], kin[k]) for k in common} for d in draws]
    rows_t = {k: gt_reward.gt_metrics(draws[0][k]["teacher"], kin[k]) for k in common}

    def field(src, key):
        return {k: float(v[key]) for k, v in src.items() if v[key] is not None}

    print(f"\n-- driver-grounded (the CoC vs what the driver did) {'-' * 52}")
    print(f"  {'row':<32} {'student':>6} {'[spread]':<16} {'teacher':>6}  "
          f"{'difference [95% CI]':<24} {'n':<6} {'McNemar p':<16}")
    grounded = []
    for key, lbl, hib in (("consistent", "GT-consistent (checkable)", True),
                          ("gt_false_clear", "GT false-clear (braked)", False),
                          ("direction_ok", "direction ok (turns)", True),
                          ("hazard_ungrounded", "hazard named, no reaction", False)):
        r = paired_row(lbl, [field(s, key) for s in rows_s], field(rows_t, key), rng)
        grounded.append((r, hib))
        print(fmt(r, hib))
        if key == "hazard_ungrounded":
            print("       ^ degenerate when paired: the value reads only the DRIVER's "
                  "reaction, so on\n         every window where both name a hazard the two "
                  "columns agree by construction.")

    # ------------------------------------------- the user's braking rows ----
    # "the driver slowed or stopped - did the CoC say so?", split by severity.
    def slow_field(src_rows, sev_filter):
        out = {}
        for k in common:
            sev = severity(kin[k])
            if sev is None or (sev_filter and sev != sev_filter):
                continue
            man = parse(src_rows[k])["maneuver"] or "NONE"
            out[k] = float(man in SLOWING)
        return out

    print(f"\n-- braked -> says slow/stop {'-' * 76}")
    braking = []
    for sev, lbl in ((None, "ANY braking window"), ("stop", "driver STOPPED"),
                     ("hard", "driver braked HARD"), ("mild", "driver slowed mildly")):
        r = paired_row(lbl, [slow_field({k: d[k]["student"] for k in common}, sev)
                             for d in draws],
                       slow_field({k: draws[0][k]["teacher"] for k in common}, sev), rng)
        braking.append(r)
        print(fmt(r, True))

    # ------------------------------------------------------- over-claim ----
    # The hedge check: on a window where the driver HELD speed, did the CoC
    # claim a speed change anyway? This is the row run 6 gamed (D-049).
    def overclaim(src_rows):
        out = {}
        for k in common:
            if kin[k]["speed"] != "holds":
                continue
            man = parse(src_rows[k])["maneuver"] or "NONE"
            out[k] = float(man in SPEED_CLAIM)
        return out

    print(f"\n-- restraint {'-' * 91}")
    r_over = paired_row("over-claim on hold-speed", [overclaim({k: d[k]["student"] for k in common})
                                                     for d in draws],
                        overclaim({k: draws[0][k]["teacher"] for k in common}), rng)
    print(fmt(r_over, False))

    # ---------------------------------------- agreement with the teacher ----
    print(f"\n-- agreement with the teacher (not correctness) {'-' * 56}")
    agree = defaultdict(list)

    def m(xs):                    # mean over the values that are defined
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else float("nan")

    for d in draws:
        sc = [d[k]["score"] for k in common if d[k]["score"]]
        agree["maneuver match"].append(m([x["maneuver_match"] for x in sc]))
        agree["direction match"].append(m([x["direction_match"] for x in sc]))
        agree["object recall"].append(m([x["obj_recall"] for x in sc]))
        agree["object precision"].append(m([x["obj_precision"] for x in sc]))
        agree["token F1"].append(m([x["token_f1"] for x in sc]))
        agree["parse rate"].append(m([x["parsed"] for x in sc]))
        agree["exact text"].append(m([d[k]["student"].strip() == d[k]["teacher"].strip()
                                      for k in common]))
    for k, v in agree.items():
        print(f"  {k:<32} {np.mean(v):6.3f} [{min(v):.3f}-{max(v):.3f}]")

    # ------------------------------------------------- joint failure ----
    # The claim "as good as the teacher" is compatible with two very different
    # worlds: independent errors that happen to balance, or the SAME errors.
    print(f"\n-- where the two disagree (draw-averaged windows) {'-' * 54}")
    for r in [g[0] for g in grounded] + braking + [r_over]:
        if not r["n"]:
            continue
        both = only_s = only_t = neither = 0.0
        for sv, tv in zip(r["s_by_win"], r["t_by_win"]):
            s = float(np.mean(sv))
            both += s * tv
            only_s += s * (1 - tv)
            only_t += (1 - s) * tv
            neither += (1 - s) * (1 - tv)
        print(f"  {r['label']:<32} both {both:6.1f}  student-only {only_s:5.1f}  "
              f"teacher-only {only_t:5.1f}  neither {neither:6.1f}   (n={r['n']})")

    # ------------------------------------------------ maneuver mixture ----
    print(f"\n-- maneuver mixture {'-' * 84}")
    s_man, t_man = Counter(), Counter()
    for d in draws:
        for k in common:
            s_man[parse(d[k]["student"])["maneuver"] or "NONE"] += 1 / len(draws)
    for k in common:
        t_man[parse(draws[0][k]["teacher"])["maneuver"] or "NONE"] += 1
    print(f"  {'maneuver':<14} {'student':>9} {'teacher':>9}   (per draw / once)")
    for m in sorted(set(s_man) | set(t_man), key=lambda m: -t_man[m]):
        print(f"  {m:<14} {s_man[m]:9.1f} {t_man[m]:9d}")

    # ------------------------------------------------------- direction ----
    # `direction_ok` scores a miss and a CONTRADICTION the same. They are not the
    # same: naming the opposite direction is the failure D-038 caught the teacher
    # in (one "turn left" in six was a right turn), and saying nothing is not.
    print(f"\n-- direction, split by how it fails {'-' * 69}")
    turns = [k for k in common if kin[k]["lateral"] != "straight"]
    straight = [k for k in common if kin[k]["lateral"] == "straight"]
    print(f"  driver turned on {len(turns)} windows, went straight on {len(straight)}")
    for who in ("student", "teacher"):
        src = draws if who == "student" else draws[:1]
        c = Counter()
        for d in src:
            for k in turns:
                gt = "left" if kin[k]["lateral"].endswith("left") else "right"
                dd = parse(d[k][who])["direction"]
                c["correct" if dd == gt else "OPPOSITE" if dd else "silent"] += 1 / len(src)
        tot = max(sum(c.values()), 1)
        print(f"  {who:<8} on the turns: " + "  ".join(
            f"{kk} {vv:.0f} ({vv / tot:.0%})" for kk, vv in c.most_common()))
    for who in ("student", "teacher"):
        src = draws if who == "student" else draws[:1]
        named = float(np.mean([sum(parse(d[k][who])["direction"] is not None for k in straight)
                               for d in src]))
        print(f"  {who:<8} names a direction on a STRAIGHT window: "
              f"{named:.0f}/{len(straight)} ({named / max(len(straight), 1):.0%})")

    # ------------------------------------------------ confusion matrix ----
    # The marginals can match while the windows do not. Rows = teacher, columns
    # = what the student said on those same windows, averaged over draws.
    print(f"\n-- what the student says where the teacher says X {'-' * 55}")
    conf = defaultdict(Counter)
    for d in draws:
        for k in common:
            tm = parse(d[k]["teacher"])["maneuver"] or "NONE"
            sm = parse(d[k]["student"])["maneuver"] or "NONE"
            conf[tm][sm] += 1 / len(draws)
    for tm in sorted(conf, key=lambda m: -sum(conf[m].values())):
        tot = sum(conf[tm].values())
        top = ", ".join(f"{sm} {c / tot:.0%}" for sm, c in conf[tm].most_common(4))
        print(f"  teacher {tm:<12} n={tot:6.1f}   student: {top}")

    # --------------------------------------------------------- verbosity ----
    s_len = [np.mean([len(d[k]["student"].split()) for k in common]) for d in draws]
    t_len = np.mean([len(draws[0][k]["teacher"].split()) for k in common])
    print(f"\n-- length (words) {'-' * 86}")
    print(f"  student {np.mean(s_len):5.1f} [{min(s_len):.1f}-{max(s_len):.1f}]"
          f"   teacher {t_len:5.1f}")

    # ------------------------------------------------ decode noise floor ----
    if len(draws) > 1:
        print(f"\n-- decode noise: draw vs draw, same checkpoint {'-' * 57}")
        pairs = [(i, j) for i in range(len(draws)) for j in range(i + 1, len(draws))]
        same_txt = [np.mean([draws[i][k]["student"].strip() == draws[j][k]["student"].strip()
                             for k in common]) for i, j in pairs]
        same_man = [np.mean([parse(draws[i][k]["student"])["maneuver"]
                             == parse(draws[j][k]["student"])["maneuver"] for k in common])
                    for i, j in pairs]
        print(f"  identical text across draws        {np.mean(same_txt):6.3f} "
              f"[{min(same_txt):.3f}-{max(same_txt):.3f}]")
        print(f"  same maneuver across draws         {np.mean(same_man):6.3f} "
              f"[{min(same_man):.3f}-{max(same_man):.3f}]")

    if a.examples:
        print(f"\n-- windows where exactly one column is right (braking) {'-' * 49}")
        r = braking[0]
        shown = {"student": 0, "teacher": 0}
        for k, sv, tv in zip(r["keys"], r["s_by_win"], r["t_by_win"]):
            s = float(np.mean(sv))
            who = "student" if (s == 1 and tv == 0) else "teacher" if (s == 0 and tv == 1) else None
            if who and shown[who] < a.examples:
                shown[who] += 1
                kk = kin[k]
                print(f"  [{who} right] {k[0]}:{k[1]}  v0={kk['v0']:.1f} dv={kk['dv']:+.1f} "
                      f"vmin={kk['vmin']:.1f}")
                print(f"      student: {draws[0][k]['student'][:150]}")
                print(f"      teacher: {draws[0][k]['teacher'][:150]}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
