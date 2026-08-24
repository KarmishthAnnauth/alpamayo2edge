"""Full-pipeline open-loop eval on the challenging split + geographic holdout.

NOT RUNNABLE YET, and deliberately left that way (D-032). Three things are wrong
with it, all of which stage 2 has to fix anyway:

  * `DistillShardDataset` returns teacher targets only — no student context, so
    `generate_traj_tokens` has no `input_ids` to prefill from. It wants
    `Stage1Dataset(..., for_generation=True)`, like `eval/coarse_minade.py`.
  * `sample_refined_trajectory` goes through `_gen_pathway_forward`, the packed
    gen-pathway forward that is still VALIDATE-ON-GPU.
  * the headline eval should free-run the CoC (decode text to `<|cot_end|>`, then
    the restricted 128-token trajectory decode) rather than teacher-force it as
    the epoch gate does. That two-phase decode does not exist yet.

`holdout_geo` additionally has no shards at all: `curation.curate` drops JPN/ZAF
before the teacher ever sees those clips. Evaluating there needs an input-only
caching pass — GT and student frames, no teacher — which is cheap but unwritten.
"""
import functools, sys
sys.path.insert(0, "src")
import torch
from torch.utils.data import DataLoader
from distill.config import load_config
from distill.data.dataset import DistillShardDataset, collate_stage2
from distill.student.edge_wrapper import EdgeStudent
from distill.eval.open_loop import evaluate
from distill.eval.coarse_minade import _split_clips

cfg = load_config()
student = EdgeStudent(cfg).cuda()  # load your stage2 checkpoint inside the wrapper
for split in ("challenging", "holdout_geo", "val"):
    ds = DistillShardDataset(cfg, clip_ids=_split_clips(cfg, split))
    dl = DataLoader(ds, batch_size=8, collate_fn=functools.partial(
        collate_stage2, pad_id=student.tokenizer.pad_token_id))
    def sample_fn(batch, k):
        # discrete plan -> diffusion refinement, k modes
        modes = []
        for _ in range(k):
            tok = student.generate_traj_tokens(batch)
            batch2 = dict(batch); batch2["traj"] = tok
            modes.append(student.sample_refined_trajectory(batch2))
        return torch.stack(modes, dim=1)
    print(split, evaluate(sample_fn, dl, k=cfg.eval.minade_k))
