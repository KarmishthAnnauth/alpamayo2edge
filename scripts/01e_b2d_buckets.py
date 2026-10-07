"""Phase 2: assign every Bench2Drive training window to a sampling bucket
(src/distill/data/b2d_buckets.py) and write `<b2d_cache>/buckets_<split>.json`,
which `train_stage2` reads when `stage2.buckets.enabled`. CPU, about a minute.

    python scripts/01e_b2d_buckets.py                 # train split of paths.b2d_cache_root
    python scripts/01e_b2d_buckets.py --split val     # a look at the val mix (never resampled)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from distill.config import load_config                              # noqa: E402
from distill.data import b2d_buckets as bk                          # noqa: E402
from distill.data import bench2drive as b2d                         # noqa: E402
from distill.data.dataset import discover_shards                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cache", default=None, help="default paths.b2d_cache_root")
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = Path(a.cache or cfg.paths.b2d_cache_root)
    shards = discover_shards(root, b2d.load_b2d_split(root, a.split))
    buckets = bk.assign(shards)
    out = root / bk.BUCKET_FILE.format(split=a.split)
    out.write_text(json.dumps({"n": len(buckets), "split": a.split, "buckets": buckets}))
    share = (cfg.stage2.raw.get("buckets") or {}).get("share")
    print(f"{root} [{a.split}]: {len(buckets)} windows -> {out}")
    print(bk.table(list(buckets.values()), share))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
