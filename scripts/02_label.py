"""Offline teacher labeling over a curated increment. Resumable."""
import argparse, json, logging, sys
from pathlib import Path
sys.path.insert(0, "src")
from distill.config import load_config
from distill.teacher.labeler import run_labeling

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, required=True)
ap.add_argument("--limit", type=int, default=None,
                help="label only the first LIMIT clips of the increment (smoke test). "
                     "The curated order is deterministic, so this is a prefix of the "
                     "same nested subset - the work is reused, not redone.")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)
cfg = load_config()
clips = json.load(open(Path(cfg.paths.cache_root) / f"curated_{a.n}.json"))
if a.limit is not None:
    clips = clips[:a.limit]
run_labeling(cfg, clips)
