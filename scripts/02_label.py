"""Offline teacher labeling over a curated increment. Resumable."""
import argparse, json, logging, sys
from pathlib import Path
sys.path.insert(0, "src")
from distill.config import load_config
from distill.teacher.labeler import run_labeling

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, required=True)
a = ap.parse_args()
logging.basicConfig(level=logging.INFO)
cfg = load_config()
clips = json.load(open(Path(cfg.paths.cache_root) / f"curated_{a.n}.json"))
run_labeling(cfg, clips)
