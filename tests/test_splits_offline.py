"""Splits: nesting, disjointness, and the challenging fallback. No torch, no GPU.

The nesting property is the one worth pinning. `curation.curate` is deterministic
per `(seed, clip_id)` so `curated_500 subset curated_2000`; if val membership were
decided by position in the list instead of by clip id, every clip would be
reassigned when the increment grew and the data-scaling curve would compare three
different validation sets.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distill.config import Cfg                                    # noqa: E402
from distill.data import splits as S                              # noqa: E402


def _cfg(val_fraction=0.2, cache_root="/tmp/unused"):
    return Cfg({"paths": {"cache_root": cache_root},
                "data": {"split": {"val_fraction": val_fraction,
                                   "holdout_countries": ["JPN", "ZAF"]}}})


IDS = [f"clip_{i:04d}" for i in range(400)]


def test_train_and_val_partition_the_set():
    sp = S.make_splits(_cfg(), IDS, stratum_fn=lambda c: "default")
    assert set(sp["train"]) | set(sp["val"]) == set(IDS)
    assert not set(sp["train"]) & set(sp["val"])


def test_val_membership_nests_across_increments():
    small, large = IDS[:100], IDS
    a = S.make_splits(_cfg(), small, stratum_fn=lambda c: "default")
    b = S.make_splits(_cfg(), large, stratum_fn=lambda c: "default")
    assert set(a["val"]) <= set(b["val"])
    assert set(a["train"]) <= set(b["train"])


def test_deterministic_across_calls_and_orderings():
    a = S.make_splits(_cfg(), IDS, stratum_fn=lambda c: "default")
    b = S.make_splits(_cfg(), list(reversed(IDS)), stratum_fn=lambda c: "default")
    assert set(a["val"]) == set(b["val"])


def test_seed_changes_the_split():
    a = S.make_splits(_cfg(), IDS, stratum_fn=lambda c: "default", seed=0)
    b = S.make_splits(_cfg(), IDS, stratum_fn=lambda c: "default", seed=1)
    assert set(a["val"]) != set(b["val"])


def test_challenging_is_the_long_tail_of_val():
    def stratum(cid):
        return "construction" if int(cid.split("_")[1]) % 2 == 0 else "default"

    sp = S.make_splits(_cfg(), IDS, stratum_fn=stratum)
    assert set(sp["challenging"]) <= set(sp["val"])
    assert sp["challenging"], "half the clips are long-tail; expected a non-empty gate"
    assert all(stratum(c) in S.LONG_TAIL for c in sp["challenging"])
    # And never from the training pool - the gate must not score trained clips.
    assert not set(sp["challenging"]) & set(sp["train"])


def test_challenging_falls_back_to_val_when_too_small():
    sp = S.make_splits(_cfg(val_fraction=0.05), IDS[:60],
                       stratum_fn=lambda c: "default")
    assert sp["challenging"] == sp["val"]


def test_load_split_names_the_fix_when_the_file_is_missing(tmp_path):
    cfg = _cfg(cache_root=str(tmp_path))
    try:
        S.load_split(cfg, "challenging")
    except FileNotFoundError as e:
        assert "01b_splits.py" in str(e)
    else:
        raise AssertionError("missing split file must raise")


def test_write_then_load_round_trips(tmp_path):
    cfg = _cfg(cache_root=str(tmp_path))
    written = S.write_splits(cfg, IDS, stratum_fn=lambda c: "night")
    for name, ids in written.items():
        assert S.load_split(cfg, name) == ids
