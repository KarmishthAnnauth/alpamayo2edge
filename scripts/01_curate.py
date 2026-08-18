"""Select the curated clip set for the next increment (500 -> 2000 -> 5000)."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, "src")
from distill.config import load_config
from distill.data.curation import curate

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, required=True, help="increment size, e.g. 500/2000/5000")
a = ap.parse_args()
from distill.data.preprocess import list_clip_ids

cfg = load_config()
all_ids = list_clip_ids(cfg)  # from the PhysicalAI-AV clip index (D-013)
labels_p = Path(cfg.paths.cache_root) / "teacher_labels_index.json"
labels = json.load(open(labels_p)) if labels_p.exists() else None  # bootstrap mode
chosen = curate(cfg, all_ids, a.n, teacher_labels=labels, seed=0)
out = Path(cfg.paths.cache_root) / f"curated_{a.n}.json"
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(chosen, open(out, "w"))
print(f"wrote {len(chosen)} clip ids -> {out}")
