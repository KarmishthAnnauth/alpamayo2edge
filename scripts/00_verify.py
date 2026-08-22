"""Phase 0: verify plan assumptions against real code BEFORE building anything.
Writes cache_root/{traj_tokenizer_spec.pt, expert_layers.pt, vocab_report.json}
which the rest of the pipeline consumes.

Runs in config-only mode (no 22GB weight download) - it needs HF auth to the
gated nvidia/Alpamayo-1.5-10B repo for config/tokenizer files only (plus the
nvidia/Cosmos-Reason2-8B tokenizer it points at), and `pip install -e ../alpamayo1.5`."""
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
print("traj tokenizer:", {k: spec[k] for k in
                          ("vocab_size", "seq_len", "future_id0", "history_id0",
                           "hist_vocab_size", "hist_seq_len", "total_bins")})
assert spec["vocab_size"] == 3000 and spec["seq_len"] == 128, \
    f"future-trajectory geometry moved: {spec['vocab_size']} bins x {spec['seq_len']} tokens " \
    "- the student's appended vocab (edge_wrapper.extend_trajectory_vocab) assumes 3000/128 (D-019)"
# The student's context carries ego motion as discrete HISTORY bins (D-029), so
# that half of the vocabulary is load-bearing too: it appends 4000 bins + 9
# structural specials and indexes them as [future | history | specials].
assert spec["hist_vocab_size"] == 1000 and spec["hist_seq_len"] == 48, \
    f"history-trajectory geometry moved: {spec['hist_vocab_size']} bins x " \
    f"{spec['hist_seq_len']} slots - prompt.N_HISTORY_SLOTS and the student's " \
    "hist_base offset assume 1000/48 (D-029)"
assert spec["total_bins"] == spec["vocab_size"] + spec["hist_vocab_size"], \
    f"the two regions do not tile traj_vocab_size ({spec['total_bins']})"
print("expert attends", len(cond["attended_layers"]), "layers (identity KV) | param:",
      cond["parameterization"], "|", cond["interpolation"])

# 0.3: tokenizer diff against the student (cross-family: Cosmos-Reason2/Qwen vs
# Nemotron), so expect vocab_ok=False and sequence-level CoC KD (D-011).
try:
    from transformers import AutoTokenizer
    edge_tok = AutoTokenizer.from_pretrained(cfg.paths.student_repo)
    report = teacher.probe_tokenizer_vs(edge_tok)
    (cache / "vocab_report.json").write_text(json.dumps(report, indent=2))
    print("vocab overlap:", report)
except Exception as e:
    print(f"vocab probe skipped (Edge tokenizer unavailable): {e}")

print("OK - proceed to 01_curate.py")
