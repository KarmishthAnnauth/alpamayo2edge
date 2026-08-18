"""Phase 0: verify plan assumptions against real code BEFORE building anything.
Writes cache_root/{traj_tokenizer_spec.pt, expert_layers.pt, vocab_report.json}
which the rest of the pipeline consumes.

Runs in config-only mode (no 34B weight download) - it needs HF auth to the
gated nvidia/Alpamayo2-Super repo for config/tokenizer files only, plus
`pip install -e ../alpamayo2`."""
import json, sys, torch
from pathlib import Path
sys.path.insert(0, "src")
from distill.config import load_config
from distill.teacher.wrapper import TeacherWrapper

cfg = load_config()
cache = Path(cfg.paths.cache_root); cache.mkdir(parents=True, exist_ok=True)
teacher = TeacherWrapper(cfg, load_model=False)  # probes need config/tokenizer only

spec = teacher.probe_trajectory_tokenizer()      # 0.2: discrete tokens still exist?
torch.save(spec, cache / "traj_tokenizer_spec.pt")
cond = teacher.probe_expert_conditioning()       # 0.2: attended layers, parameterization
torch.save(cond, cache / "expert_layers.pt")
print("traj tokenizer:", {k: spec[k] for k in ("vocab_size", "seq_len", "future_id0")})
print("expert attends", len(cond["attended_layers"]), "layers (identity KV) | param:",
      cond["parameterization"], "|", cond["interpolation"])

# 0.3: tokenizer diff against the student. Runs once the Edge tokenizer is
# available (blocked on D-009 / Cosmos 3 Edge access).
try:
    from transformers import AutoTokenizer
    edge_tok = AutoTokenizer.from_pretrained(cfg.paths.student_repo)
    report = teacher.probe_tokenizer_vs(edge_tok)
    (cache / "vocab_report.json").write_text(json.dumps(report, indent=2))
    print("vocab overlap:", report)
except Exception as e:
    print(f"vocab probe skipped (Edge tokenizer unavailable): {e}")

print("OK - proceed to 01_curate.py")
