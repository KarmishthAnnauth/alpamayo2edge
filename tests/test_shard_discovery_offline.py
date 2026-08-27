"""Shard discovery must not return the student-input files. No GPU.

`frames.save_window_input` writes `{idx:02d}_input.npz` into the same clip
directory as the teacher targets `{idx:02d}.npz`, and `data.cache_student_frames`
is on, so every labeled clip directory holds both kinds. A bare `*.npz` glob
returns both: the dataset length doubles, and `Stage1Dataset.__getitem__` parses
the window index off the filename, so `int("00_input")` raises before the first
forward pass. `scripts/check_cache.py` has always partitioned on this suffix;
these pin that discovery agrees with it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distill.data.dataset import discover_shards  # noqa: E402


def _clip(root: Path, clip_id: str, windows=(0, 1), inputs=True) -> None:
    d = root / clip_id
    d.mkdir(parents=True, exist_ok=True)
    for w in windows:
        (d / f"{w:02d}.npz").write_bytes(b"")
        if inputs:
            (d / f"{w:02d}_input.npz").write_bytes(b"")


def test_input_files_are_not_returned_as_shards(tmp_path):
    _clip(tmp_path, "clip_a")
    got = discover_shards(tmp_path, ["clip_a"])
    assert [p.name for p in got] == ["00.npz", "01.npz"]


def test_window_index_parses_off_every_returned_shard(tmp_path):
    """The failure mode the filter exists to prevent, stated directly."""
    _clip(tmp_path, "clip_a")
    for p in discover_shards(tmp_path, ["clip_a"]):
        int(p.stem)          # Stage1Dataset.__getitem__ does exactly this


def test_length_is_windows_not_files(tmp_path):
    _clip(tmp_path, "clip_a")
    _clip(tmp_path, "clip_b")
    assert len(discover_shards(tmp_path, ["clip_a", "clip_b"])) == 4


def test_still_works_on_a_cache_with_no_student_inputs(tmp_path):
    """Caches written before `data/frames.py` existed have targets only."""
    _clip(tmp_path, "clip_a", inputs=False)
    assert len(discover_shards(tmp_path, ["clip_a"])) == 2


def test_manifest_drives_discovery_when_no_clip_ids_given(tmp_path):
    import json

    _clip(tmp_path, "clip_a")
    _clip(tmp_path, "clip_b")
    (tmp_path / "manifest.json").write_text(json.dumps({"clips": ["clip_a"]}))
    got = discover_shards(tmp_path)
    assert {p.parent.name for p in got} == {"clip_a"}
    assert len(got) == 2
