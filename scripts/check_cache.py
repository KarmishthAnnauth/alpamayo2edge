"""Verify every cached shard actually loads. Deletes truncated ones with --fix.

Resume keys off `path.exists()`, so a shard truncated by a kill mid-write is
worse than a missing one: it is skipped forever and only surfaces much later as
a stage-1 loader error, long after the run that produced it. Writes are atomic
as of D-031, but any shard written before that (or by an interrupted older run)
is worth checking once.

    python scripts/check_cache.py            # report
    python scripts/check_cache.py --fix      # delete the bad ones so a re-run redoes them
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from distill.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix", action="store_true",
                    help="delete unreadable shards so the next run regenerates them")
    a = ap.parse_args()

    cache_root = Path(load_config().paths.cache_root)
    shards = sorted(cache_root.glob("*/*.npz"))
    bad: list[tuple[Path, str]] = []
    for f in shards:
        try:
            with np.load(f, allow_pickle=False) as d:
                for k in d.files:      # touch every member: the zip index alone
                    _ = d[k].shape     # can be intact while the payload is not
        except Exception as e:
            bad.append((f, f"{type(e).__name__}: {str(e)[:80]}"))

    targets = sum(1 for f in shards if not f.name.endswith("_input.npz"))
    print(f"{cache_root}")
    print(f"  {len(shards)} files ({targets} targets, {len(shards) - targets} inputs)")
    print(f"  unreadable: {len(bad)}")
    for f, msg in bad:
        print(f"    {f.relative_to(cache_root)}  {msg}")
        if a.fix:
            f.unlink()
    if bad and a.fix:
        print(f"  deleted {len(bad)}; re-run 02_label.py to regenerate them")
    elif bad:
        print("  re-run with --fix to delete them, then re-run 02_label.py")
    return 1 if bad and not a.fix else 0


if __name__ == "__main__":
    raise SystemExit(main())
