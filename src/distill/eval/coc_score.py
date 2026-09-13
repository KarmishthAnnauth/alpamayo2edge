"""Structured scoring for chain-of-causation text.

`05a_inspect_coc.py` prints 8 windows for a human to read; `val CoC NLL` is a
teacher-forced token-level CE. Neither answers "does the student's OWN reasoning
say the same thing as the teacher's", which is the question that matters once the
CoC is what conditions stage 2 (the action expert attends the reasoner's KV, not
the trajectory tokens).

This module scores that, and it can because the teacher's CoC is a near-regular
language. Measured over the 19930 non-empty cached traces:

    94.6%  contain a causal connective  ->  <action> <connective> <cause>
    95.8%  are a single clause
    99.9%  of leading tokens come from a 25-word maneuver vocabulary
    96.2%  name at least one object from a ~20-noun inventory

So the parse below is not a heuristic hoping for the best - it is a grammar the
corpus actually follows, and `parse_rate` in the report says how often it held.

NOT a semantic judge. It scores maneuver class, direction and named objects.
Two traces that agree on all three can still differ in meaning, and the metric
cannot see it - which is why `05a_inspect_coc.py` stays.
"""
from __future__ import annotations
import re

CONNECTIVE = re.compile(
    r"\b(since|due to|because of|because|as|to avoid|in order to|so that|"
    r"while|after|before|for)\b", re.I)

# Ordered, first match wins: specific phrases before the bare verbs they contain,
# so "keep distance" lands on FOLLOW rather than KEEP.
MANEUVER_RULES: list[tuple[str, str]] = [
    ("STOP",        r"\b(stop|halt|come to a stop|wait)\b"),
    ("YIELD",       r"\b(yield|give way)\b"),
    ("FOLLOW",      r"\b(keep|maintain|create|increase)\s+(a\s+)?(safe\s+)?(distance|gap|following)"
                    r"|\bfollow\b|\bgap-search\b"),
    ("LANE_CHANGE", r"\b(lane change|change lanes?|merge|move (in)?to the (left|right))\b"),
    ("TURN",        r"\b(turn|make a (left|right))\b"),
    ("NUDGE",       r"\b(nudge|shift|drift|steer|edge)\b"),
    ("ACCELERATE",  r"\b(accelerate|speed up|resume|increase speed)\b"),
    ("SLOW",        r"\b(slow|decelerate|brake|reduce speed|ease off)\b"),
    ("ADAPT_SPEED", r"\b(adapt|adjust|match)\b"),
    ("KEEP",        r"\b(keep|maintain|remain|stay|continue|proceed|split)\b"),
]

DIRECTIONS = ("left", "right", "straight")

# Object inventory, collapsed to the classes a driving decision turns on.
OBJECTS: dict[str, str] = {
    "VEHICLE":      r"\b(vehicles?|cars?|trucks?|buses|bus|vans?|traffic)\b",
    "VRU":          r"\b(pedestrians?|cyclists?|bikes?|bicycles?|motorcycles?|scooters?)\b",
    "SIGNAL":       r"\b(traffic light|lights?|signals?|signalized)\b",
    "SIGN":         r"\b(signs?|stop sign)\b",
    "CONSTRUCTION": r"\b(cones?|barriers?|barricades?|construction|roadworks?)\b",
    "JUNCTION":     r"\b(intersections?|junctions?|crosswalks?|roundabouts?)\b",
}
# Things whose presence the ego must react to. Used by the false-clear check:
# LANE/JUNCTION/SIGN are scene furniture, not obstacles.
HAZARDS = ("VEHICLE", "VRU", "CONSTRUCTION")
# The student asserting open road.
CLEAR = re.compile(r"\b(clear|empty|unobstructed|no (other )?(vehicles?|traffic|obstacles?))\b", re.I)
# Maneuvers that mean "carry on" - a false-clear only counts if the student also acts on it.
PROCEEDING = ("KEEP", "ACCELERATE")


def split_clause(text: str) -> tuple[str, str]:
    """`<action>` before the first causal connective, `<cause>` after it.

    No connective -> the whole trace is the action and the cause is empty, which
    is the right reading for the 5.4% that are a bare imperative."""
    m = CONNECTIVE.search(text)
    if not m:
        return text.strip(), ""
    return text[:m.start()].strip(), text[m.end():].strip()


def maneuver(action: str) -> str | None:
    for name, pat in MANEUVER_RULES:
        if re.search(pat, action, re.I):
            return name
    return None


def direction(action: str) -> str | None:
    """Direction of the EGO maneuver, so it is read off the action clause only -
    'the vehicle on our left' in a cause is a location, not a maneuver."""
    for d in DIRECTIONS:
        if re.search(rf"\b{d}\b", action, re.I):
            return d
    return None


def objects(text: str) -> set[str]:
    return {c for c, pat in OBJECTS.items() if re.search(pat, text, re.I)}


def parse(text: str) -> dict:
    action, cause = split_clause(text.strip())
    return {"text": text.strip(), "action": action, "cause": cause,
            "maneuver": maneuver(action), "direction": direction(action),
            "objects": objects(text)}


def token_f1(a: str, b: str) -> float:
    """Bag-of-words F1 - a coarse phrasing number next to the structured ones.

    Deliberately not BLEU/ROUGE: these traces are ~12 words, where n-gram metrics
    are dominated by the shared template ('... since it is ... ahead in our lane')
    and barely move on the content words that decide the maneuver."""
    ta = re.findall(r"[a-z]+", a.lower())
    tb = re.findall(r"[a-z]+", b.lower())
    if not ta or not tb:
        return 0.0
    from collections import Counter
    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if not overlap:
        return 0.0
    p, r = overlap / len(ta), overlap / len(tb)
    return 2 * p * r / (p + r)


def score_pair(student: str, teacher: str) -> dict | None:
    """One (student, teacher) CoC pair -> per-window fields. None if the teacher
    trace is empty (0.3% of the cache), which is unscoreable rather than wrong."""
    if not teacher.strip():
        return None
    s, t = parse(student), parse(teacher)
    s_obj, t_obj = s["objects"], t["objects"]
    inter = s_obj & t_obj
    t_haz = t_obj & set(HAZARDS)
    return {
        "maneuver_match": (s["maneuver"] == t["maneuver"]) if t["maneuver"] else None,
        "student_maneuver": s["maneuver"], "teacher_maneuver": t["maneuver"],
        "direction_match": (s["direction"] == t["direction"]) if t["direction"] else None,
        "obj_recall": len(inter) / len(t_obj) if t_obj else None,
        "obj_precision": len(inter) / len(s_obj) if s_obj else None,
        # The safety-asymmetric one: the teacher saw something to react to, the
        # student named none of it AND chose to carry on. Window 1/7 of the
        # run-313 inspection ("Keep lane since the lane is clear ahead" against a
        # cut-in and a lead vehicle) is exactly this.
        "false_clear": bool(t_haz) and not (s_obj & set(HAZARDS))
                       and (s["maneuver"] in PROCEEDING or bool(CLEAR.search(student))),
        "hazard_window": bool(t_haz),
        "token_f1": token_f1(student, teacher),
        "parsed": s["maneuver"] is not None and t["maneuver"] is not None,
    }
