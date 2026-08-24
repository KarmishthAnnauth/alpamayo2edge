"""The fetch pool must overlap streaming with the teacher without reordering
the cache or breaking free resume.

Labeling is two bottlenecks in series: `load_window` is network-bound (measured
7-17 s per camera, four cameras) and `label_window` is GPU-bound. Run serially
the GPU idles ~80% of the time — a live 500-clip run measured 30 s/window at 20%
utilisation. Fetching several windows at once is the whole speedup, and NVIDIA's
`load_physical_aiavdataset` fetches its cameras serially inside one call, so the
concurrency has to be at the window level.

Three things can go wrong, none of which shows up as an exception:

* **Resume regresses.** If work is submitted before the filesystem is consulted,
  the pool cheerfully streams every window the run is about to skip — the exact
  cost `plan_work` exists to avoid, and worse with a pool than without.
* **Order drifts.** Shards are keyed (clip_id, w_idx) so the cache survives it,
  but out-of-order consumption unbounds memory: decoded windows pile up waiting
  for a slow head instead of being capped at `depth`.
* **No actual concurrency.** A pool that silently serialises still passes every
  correctness test and just runs at the old speed.
"""
import sys
import threading
import time
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.data import frames  # noqa: E402
from distill.teacher import labeler  # noqa: E402


class _Cfg(dict):
    def __getattr__(self, k):
        v = self[k]
        return _Cfg(v) if isinstance(v, dict) else v

    def get(self, k, default=None):
        v = dict.get(self, k, default)
        return _Cfg(v) if isinstance(v, dict) else v

    @property
    def raw(self):
        return {}


def _cfg(root, workers, depth=None):
    teacher = dict(flow_targets_per_window=12, topk_logits=32, max_coc_tokens=48,
                   n_traj_samples=4, prefetch_workers=workers)
    if depth is not None:
        teacher["prefetch_depth"] = depth
    return _Cfg(paths=_Cfg(cache_root=str(root)), eval=_Cfg(horizon_s=6.4),
                data=_Cfg(windows_per_clip=2, cache_student_frames=True),
                teacher=_Cfg(teacher))


CLIPS = [f"clip{i:02d}" for i in range(6)]        # 6 clips x 2 windows = 12
FETCH_S, TEACHER_S = 0.04, 0.02


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Counts concurrency in `load_window` and records consumption order."""
    st = types.SimpleNamespace(inflight=0, peak=0, loaded=[], labeled=[],
                               lock=threading.Lock())

    def fake_load_window(cfg, clip_id, t0_us):
        with st.lock:
            st.inflight += 1
            st.peak = max(st.peak, st.inflight)
            st.loaded.append(clip_id)
        time.sleep(FETCH_S)
        with st.lock:
            st.inflight -= 1
        return types.SimpleNamespace(clip_id=clip_id)

    def fake_label(window, **kw):
        time.sleep(TEACHER_S)
        st.labeled.append(window.clip_id)
        return types.SimpleNamespace()

    def touch(p, text="x"):
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    monkeypatch.setattr(labeler, "load_window", fake_load_window)
    monkeypatch.setattr(labeler, "save_shard", lambda path, out: touch(path, "shard"))
    monkeypatch.setattr(labeler, "TeacherWrapper",
                        lambda cfg: types.SimpleNamespace(label_window=fake_label))
    monkeypatch.setattr(frames, "save_window_input",
                        lambda p, w, quality=92: touch(p, "in"))

    def run(workers, preexisting=(), depth=None):
        for clip_id, w_idx, kind in preexisting:
            touch(labeler.shard_path(tmp_path, clip_id, w_idx) if kind == "shard"
                  else frames.input_path(tmp_path, clip_id, w_idx))
        st.inflight = st.peak = 0
        st.loaded.clear()
        st.labeled.clear()
        t = time.perf_counter()
        labeler.run_labeling(_cfg(tmp_path, workers, depth), list(CLIPS))
        st.wall = time.perf_counter() - t
        return st

    return run


def test_pool_actually_fetches_concurrently(harness):
    """A pool that silently serialises passes every correctness test below and
    just runs at the old speed. This is the one that catches that."""
    st = harness(workers=4)
    assert st.peak >= 2, f"no concurrency: peak in-flight was {st.peak}"
    assert len(st.labeled) == 12


def test_single_worker_is_strictly_serial(harness):
    """workers<=1 takes the plain path — the one to debug on, since a traceback
    from a pool thread loses the enclosing context."""
    st = harness(workers=1)
    assert st.peak == 1
    assert len(st.labeled) == 12


def test_consumption_order_matches_the_curated_list(harness):
    """Out-of-order completion is fine for the cache but unbounds memory, since
    decoded windows would pile up behind a slow head instead of capping at
    `depth`."""
    st = harness(workers=4)
    assert st.labeled == [c for c in CLIPS for _ in range(2)]


def test_prefetch_overlaps_fetch_with_teacher(harness):
    """The point of the exercise: wall clock beats the serial sum."""
    serial = 12 * (FETCH_S + TEACHER_S)
    st = harness(workers=4)
    assert st.wall < serial * 0.8, f"no overlap: {st.wall:.2f}s vs serial {serial:.2f}s"


def test_resume_still_streams_nothing(harness):
    """THE REGRESSION, now with a pool in front of it. Submitting before
    consulting the filesystem would stream every window the run is about to
    skip — the cost `plan_work` exists to avoid, and worse with a pool."""
    cached = [(c, w, k) for c in CLIPS for w in (0, 1) for k in ("shard", "input")]
    st = harness(workers=4, preexisting=cached)
    assert st.loaded == []
    assert st.labeled == []


def test_partial_resume_streams_only_what_is_missing(harness):
    cached = [(c, w, k) for c in CLIPS[:4] for w in (0, 1) for k in ("shard", "input")]
    st = harness(workers=4, preexisting=cached)
    assert set(st.loaded) == set(CLIPS[4:])
    assert len(st.loaded) == 4


def test_depth_bounds_windows_in_flight(harness):
    """`depth` is the memory knob: each in-flight window holds decoded frames."""
    st = harness(workers=8, depth=2)
    assert st.peak <= 2, f"depth ignored: {st.peak} in flight"


def test_plan_work_touches_no_network(tmp_path, monkeypatch):
    """`plan_work` is a filesystem scan and must stay one."""
    monkeypatch.setattr(labeler, "load_window",
                        lambda *a, **k: pytest.fail("plan_work streamed a window"))
    work, skipped = labeler.plan_work(_cfg(tmp_path, 4), tmp_path, list(CLIPS), True)
    assert len(work) == 12 and skipped == 0
    assert [w.clip_id for w in work] == [c for c in CLIPS for _ in range(2)]


def test_shard_write_is_atomic(tmp_path):
    """`sbatch_label.sh` claimed atomic writes; np.savez_compressed wrote straight
    to the destination. Resume keys off existence, so a shard truncated by a kill
    mid-write is worse than a missing one - skipped forever, surfacing later as a
    loader error long after the run that made it. A 3-day 5000-clip run has ample
    opportunity to be interrupted."""
    import numpy as np

    labeler.atomic_savez(tmp_path / "00.npz", True, a=np.arange(5))
    assert np.load(tmp_path / "00.npz")["a"].tolist() == [0, 1, 2, 3, 4]

    with pytest.raises(Exception):
        labeler.atomic_savez(tmp_path / "01.npz", True, bad=lambda x: x)
    assert not (tmp_path / "01.npz").exists(), "truncated destination survived"
    assert not [f for f in tmp_path.iterdir() if f.name.endswith(".tmp.npz")]


def _failing_harness(tmp_path, monkeypatch, fail_for, fail_teacher=()):
    """Like `harness`, but `load_window` raises for clips in `fail_for` and
    `label_window` raises for clips in `fail_teacher`."""
    st = types.SimpleNamespace(loaded=[], labeled=[])

    def touch(p, text="x"):
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def fake_load_window(cfg, clip_id, t0_us):
        if clip_id in fail_for:
            raise ConnectionError(f"simulated fetch failure for {clip_id}")
        st.loaded.append(clip_id)
        return types.SimpleNamespace(clip_id=clip_id)

    def fake_label(window, **kw):
        if window.clip_id in fail_teacher:
            raise RuntimeError(f"simulated teacher failure for {window.clip_id}")
        st.labeled.append(window.clip_id)
        return types.SimpleNamespace()

    monkeypatch.setattr(labeler, "load_window", fake_load_window)
    monkeypatch.setattr(labeler, "save_shard", lambda path, out: touch(path, "shard"))
    monkeypatch.setattr(labeler, "TeacherWrapper",
                        lambda cfg: types.SimpleNamespace(label_window=fake_label))
    monkeypatch.setattr(frames, "save_window_input",
                        lambda p, w, quality=92: touch(p, "in"))
    return st


def test_a_failed_fetch_does_not_kill_the_run(tmp_path, monkeypatch):
    """THE UNATTENDED-RUN REGRESSION. huggingface_hub retries 5x; a fetch that
    fails past that used to propagate out of fut.result() and end the whole pass.
    Over a 27-hour run on a network we have watched retry, that is the difference
    between a finished cache and a stack trace from 2am."""
    st = _failing_harness(tmp_path, monkeypatch, fail_for={"clip02"})
    labeler.run_labeling(_cfg(tmp_path, 4), list(CLIPS))
    assert set(st.labeled) == set(CLIPS) - {"clip02"}
    assert len(st.labeled) == 10          # 5 surviving clips x 2 windows


def test_failed_windows_leave_no_shard_so_a_rerun_retries_them(tmp_path, monkeypatch):
    st = _failing_harness(tmp_path, monkeypatch, fail_for={"clip02"})
    labeler.run_labeling(_cfg(tmp_path, 4), list(CLIPS))
    assert not labeler.shard_path(tmp_path, "clip02", 0).exists()
    assert (tmp_path / "failed_windows.json").exists()

    # Second pass with the network "healed": only the failed windows are redone.
    st2 = _failing_harness(tmp_path, monkeypatch, fail_for=set())
    labeler.run_labeling(_cfg(tmp_path, 4), list(CLIPS))
    assert set(st2.labeled) == {"clip02"}


def test_a_failed_teacher_pass_is_also_survived(tmp_path, monkeypatch):
    st = _failing_harness(tmp_path, monkeypatch, fail_for=set(),
                          fail_teacher={"clip03"})
    labeler.run_labeling(_cfg(tmp_path, 4), list(CLIPS))
    assert set(st.labeled) == set(CLIPS) - {"clip03"}


def test_systemic_failure_trips_the_circuit_breaker(tmp_path, monkeypatch):
    """Scattered failures are the network; a solid run of them is an expired
    token or a full disk, and grinding through thousands of windows writing
    nothing is worse than stopping."""
    st = _failing_harness(tmp_path, monkeypatch, fail_for=set(CLIPS))
    cfg = _cfg(tmp_path, 4)
    cfg["teacher"]["max_consecutive_failures"] = 3
    labeler.run_labeling(cfg, list(CLIPS))
    import json as _json
    recorded = _json.loads((tmp_path / "failed_windows.json").read_text())
    assert len(recorded) == 3, f"breaker did not stop at 3: {len(recorded)}"
