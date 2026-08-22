"""The student context must mirror the teacher's byte-for-byte where it can.

These tests pin the format against the two NVIDIA sources it was copied from
(`alpamayo1.5/helper.py::create_message` and `alpamayo-recipes`'
`chat_template/components.py`). If someone edits a string in prompt.py, the
expected literal here has to be edited too — which is the point: the format is
not ours to drift.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.student import prompt  # noqa: E402

CAMS = ["camera_cross_left_120fov", "camera_front_wide_120fov",
        "camera_cross_right_120fov", "camera_front_tele_30fov"]


def test_user_context_matches_the_teacher_format():
    got = prompt.render(prompt.user_segments(CAMS, n_frames=2), image_token="<I>")
    expect = (
        "Front left camera: frame 0 <I>frame 1 <I>"
        "Front camera: frame 0 <I>frame 1 <I>"
        "Front right camera: frame 0 <I>frame 1 <I>"
        "Front telephoto camera: frame 0 <I>frame 1 <I>"
        "<|traj_history_start|>" + "<|traj_history|>" * 48 + "<|traj_history_end|>"
        "output the chain-of-thought reasoning of the driving process, "
        "then output the future trajectory."
    )
    assert got == expect


def test_cameras_are_ordered_by_index_not_by_input_order():
    """construct_image asserts ascending camera ids; the model learned that order."""
    shuffled = ["camera_front_tele_30fov", "camera_cross_right_120fov",
                "camera_front_wide_120fov", "camera_cross_left_120fov"]
    assert prompt.render(prompt.user_segments(shuffled, 1)) == \
           prompt.render(prompt.user_segments(CAMS, 1))


def test_route_is_inserted_between_history_and_instruction():
    got = prompt.render(prompt.user_segments(CAMS[:1], 1, nav_text="Turn left in 40m"))
    assert "<|traj_history_end|><|route_start|>Turn left in 40m<|route_end|>output the chain" in got


def test_unknown_camera_is_rejected():
    with pytest.raises(ValueError, match="unknown camera"):
        prompt.user_segments(["camera_made_up"], 1)


def test_generation_mode_opens_cot_and_stops():
    """Matches create_message, whose assistant turn is exactly '<|cot_start|>'."""
    assert prompt.render(prompt.assistant_segments()) == "<|cot_start|>"


def test_training_target_is_cot_then_trajectory():
    segs = prompt.assistant_segments(coc_text="slowing for the cyclist",
                                     traj_bins=list(range(128)))
    assert prompt.render(segs) == (
        "<|cot_start|>slowing for the cyclist<|cot_end|>"
        "<|traj_future_start|><bin>x128<|traj_future_end|>")


# --- assemble ---------------------------------------------------------------

def _stub():
    """A tokenizer stand-in: one id per character, specials and bins in their
    own high ranges so spans are unambiguous."""
    return dict(
        encode=lambda s: [ord(c) % 50 for c in s],
        special_id=lambda t: 900 + prompt.SPECIAL_TOKENS.index(t),
        bin_id=lambda b: 10_000 + b,
        image_tokens=lambda i: 4,
        image_token_id=777,
    )


def test_assemble_marks_the_ego_history_span():
    a = prompt.assemble(prompt.user_segments(CAMS[:1], n_frames=1), **_stub())
    lo, hi = a.history_span
    assert hi - lo == prompt.N_HISTORY_SLOTS
    assert set(a.input_ids[lo:hi]) == {900 + prompt.SPECIAL_TOKENS.index("<|traj_history|>")}


def test_assemble_marks_one_span_per_frame_in_prompt_order():
    a = prompt.assemble(prompt.user_segments(CAMS, n_frames=3), **_stub())
    assert len(a.image_spans) == 4 * 3
    for lo, hi in a.image_spans:
        assert hi - lo == 4
        assert set(a.input_ids[lo:hi]) == {777}


def test_assemble_marks_coc_and_trajectory_spans():
    segs = prompt.assistant_segments(coc_text="abc", traj_bins=list(range(128)))
    a = prompt.assemble(segs, **_stub())
    assert a.coc_span is not None and a.traj_span is not None
    assert a.input_ids[a.coc_span[0]:a.coc_span[1]] == [ord(c) % 50 for c in "abc"]
    lo, hi = a.traj_span
    assert hi - lo == 128
    assert a.input_ids[lo] == 10_000 and a.input_ids[hi - 1] == 10_127


def test_assemble_generation_mode_has_no_trajectory_span():
    a = prompt.assemble(prompt.assistant_segments(), **_stub())
    assert a.traj_span is None and a.coc_span is None


def test_assemble_rejects_a_short_trajectory():
    segs = prompt.assistant_segments(coc_text="x", traj_bins=[1, 2, 3])
    with pytest.raises(ValueError, match="expected 128 trajectory bins"):
        prompt.assemble(segs, **_stub())


def test_special_tokens_cover_every_marker_the_template_emits():
    """Any marker in the rendered context must be a token we actually append."""
    text = prompt.render(prompt.user_segments(CAMS, 1, nav_text="x")) + \
        prompt.render(prompt.assistant_segments("c", list(range(128))))
    import re
    for marker in set(re.findall(r"<\|[a-z_]+\|>", text)):
        assert marker in prompt.SPECIAL_TOKENS, f"{marker} is emitted but never appended"
