"""Curated clip selection for the 5k single-GPU profile.

Strategy (plan v2): stratified sampling that oversamples long-tail scenarios,
with a hard geographic holdout. Works in two modes:
  - metadata mode: uses fields available in the PhysicalAI-AV clip JSON
    (country, time-of-day, weather where present);
  - bootstrap mode: after a first 500-clip labeling pass, teacher meta-action
    labels refine the strata (e.g. clips whose teacher meta-action is
    yield / lane_change / stop are long-tail-ish and get upweighted).

Deterministic given the seed, so the 500 -> 2000 -> 5000 increments are nested:
every smaller set is a strict subset of every larger one, which is what makes
the scaling curve an apples-to-apples comparison.
"""
from __future__ import annotations
import random

from .preprocess import clip_metadata

LONGTAIL_META_ACTIONS = {"yield", "stop", "lane_change", "overtake"}


def stratum_of(meta: dict, teacher_labels: dict | None) -> str:
    """Map a clip to the first matching stratum name from the config weights."""
    tags = {str(t).lower() for t in meta.get("tags", [])}
    weather = str(meta.get("weather", "")).lower()
    tod = str(meta.get("time_of_day", "")).lower()
    if teacher_labels and teacher_labels.get("meta_action") in LONGTAIL_META_ACTIONS:
        return "cut_in" if teacher_labels["meta_action"] == "overtake" else "intersection"
    if "construction" in tags:
        return "construction"
    if "pedestrian" in tags or "crosswalk" in tags:
        return "pedestrian"
    if "intersection" in tags or "junction" in tags:
        return "intersection"
    if weather in {"rain", "snow", "fog", "heavy_rain"}:
        return "adverse_weather"
    if tod in {"night", "dusk", "dawn"}:
        return "night"
    return "default"


def curate(cfg, all_clip_ids: list[str], n: int,
           teacher_labels: dict[str, dict] | None = None,
           seed: int = 0) -> list[str]:
    holdout = set(cfg.data.split.raw["holdout_countries"])
    weights = cfg.data.curation.strata_weights.raw
    rng = random.Random(seed)

    weighted: list[tuple[str, float]] = []
    for cid in all_clip_ids:
        meta = clip_metadata(cfg, cid)  # data_collection.parquet row, lower-cased keys
        if meta.get("country") in holdout:
            continue  # geographic test holdout - never enters training pool
        s = stratum_of(meta, (teacher_labels or {}).get(cid))
        weighted.append((cid, float(weights.get(s, weights["default"]))))

    # Weighted sampling without replacement via exponential-sort trick;
    # deterministic per (seed, clip_id) so increments are nested supersets.
    def key(item):
        cid, w = item
        r = random.Random(f"{seed}:{cid}").random()
        return -(r ** (1.0 / max(w, 1e-6)))  # larger w -> more likely selected

    ranked = sorted(weighted, key=key)
    chosen = [cid for cid, _ in ranked[:n]]
    rng.shuffle(chosen)
    return chosen
