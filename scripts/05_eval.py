"""Full-pipeline open-loop eval on the challenging split + geographic holdout."""
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
