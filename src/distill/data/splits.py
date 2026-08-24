"""Train / val / challenging / geographic-holdout splits over a curated set.

Nothing wrote `split_*.json` until now, and three consumers read it:
`train_stage1`'s epoch gate (`eval/coarse_minade.py`), `scripts/05_eval.py` and
`scripts/07_measure_gap.py`. The gate runs at the END of epoch 1, so the missing
file surfaced hours into a run rather than before it.

**Membership is a property of the clip id, not of its position in a list.**
`curation.curate` is deterministic per `(seed, clip_id)` so the increments nest
(`curated_500 subset curated_2000 subset curated_5000`); splitting by a per-clip
hash keeps that property downstream — `val_500 subset val_2000` — which is what
makes the data-scaling curve apples-to-apples. Splitting by index would reshuffle
every clip's assignment each time the increment grows.

The geographic holdout (JPN/ZAF) is written for completeness but is NOT labeled:
`curation.curate` drops those clips before the teacher ever sees them. Evaluating
on it needs a separate input-only caching pass — teacher targets are not required
for a minADE number, only student frames and GT.
"""
from __future__ import annotations
import json
import logging
import random
from pathlib import Path

log = logging.getLogger(__name__)

#: Strata that count as long-tail. `curation.stratum_of` returns "default" for
#: everything else, and averaged minADE over easy driving saturates early at this
#: data scale (eval/open_loop.py's own docstring) — hence a hard-sample gate.
LONG_TAIL = ("construction", "pedestrian", "intersection", "adverse_weather",
             "night", "cut_in")

#: Below this the challenging split is too small to early-stop on, and the gate
#: falls back to the full val split rather than reading noise.
MIN_CHALLENGING = 5


def in_val(clip_id: str, val_fraction: float, seed: int = 0) -> bool:
    """Deterministic per-clip val membership. Independent of increment size."""
    return random.Random(f"split:{seed}:{clip_id}").random() < val_fraction


def make_splits(cfg, clip_ids: list[str], stratum_fn=None,
                seed: int = 0) -> dict[str, list[str]]:
    """`{"train": [...], "val": [...], "challenging": [...]}` over `clip_ids`.

    `stratum_fn(clip_id) -> str` is injected so this is testable without the
    dataset index; the default reads `data_collection.parquet` through
    `curation.stratum_of`, exactly as curation itself does.
    """
    if stratum_fn is None:
        stratum_fn = _default_stratum_fn(cfg)

    frac = float(cfg.data.split.val_fraction)
    val = [c for c in clip_ids if in_val(c, frac, seed)]
    train = [c for c in clip_ids if c not in set(val)]
    challenging = [c for c in val if stratum_fn(c) in LONG_TAIL]

    if len(challenging) < MIN_CHALLENGING:
        log.warning(
            "only %d long-tail clips in a val split of %d — the challenging gate "
            "would early-stop on noise, so it falls back to the full val split. "
            "Raise data.split.val_fraction or grow the increment.",
            len(challenging), len(val))
        challenging = list(val)
    return {"train": train, "val": val, "challenging": challenging}


def _default_stratum_fn(cfg):
    from .curation import stratum_of
    from .preprocess import clip_metadata
    return lambda cid: stratum_of(clip_metadata(cfg, cid), None)


def write_splits(cfg, clip_ids: list[str], stratum_fn=None,
                 seed: int = 0) -> dict[str, list[str]]:
    """Write `<cache_root>/split_<name>.json` for every split. Returns them."""
    splits = make_splits(cfg, clip_ids, stratum_fn=stratum_fn, seed=seed)
    root = Path(cfg.paths.cache_root)
    root.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (root / f"split_{name}.json").write_text(json.dumps(ids, indent=1))
    return splits


def load_split(cfg, name: str) -> list[str]:
    p = Path(cfg.paths.cache_root) / f"split_{name}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist. Run `python scripts/01b_splits.py --n <increment>` "
            "— stage-1 training reads split_train.json and its epoch gate reads "
            "split_challenging.json, so a run without them dies at the first eval.")
    return json.loads(p.read_text())
