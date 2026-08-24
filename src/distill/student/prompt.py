"""The student's deployment context — a deliberate mirror of the teacher's input.

Decision (2026-08-22): the student sees **exactly what the teacher sees** —
cameras, ego motion, and a short text instruction — and produces CoC then
trajectory tokens. This module holds that format in one place.

Every string here is copied verbatim from two independent sources that agree,
so the format is not a guess:

- `../alpamayo1.5/src/alpamayo1_5/helper.py::create_message` — the RELEASED
  inference path, which `teacher/wrapper.py::_prepare_inputs` already uses for
  labeling.
- `../alpamayo-recipes/src/alpamayo/chat_template/{r1,r1_5}.py` + `components.py`
  — the TRAINING-time template, with `components_order` pinned by
  `recipes/alpamayo1_5_sft/configs/vla_processor/default.yaml` and by
  `tests/test_recipe_static_contracts.py`.

The order is `image -> traj_history -> [route] -> prompt`, then the assistant
turn opens with `<|cot_start|>`:

    system     You are a driving assistant that generates safe and accurate actions.
    user       "Front left camera: " "frame 0 " <img> "frame 1 " <img> ...
               "Front camera: "      "frame 0 " <img> ...
               <|traj_history_start|> <|traj_history|> x48 <|traj_history_end|>
               output the chain-of-thought reasoning of the driving process,
               then output the future trajectory.
    assistant  <|cot_start|> ... <|cot_end|>
               <|traj_future_start|> <bin> x128 <|traj_future_end|>

WHAT IS DELIBERATELY NOT DECIDED HERE
-------------------------------------
Two things are the student's own and must not be copied from the teacher:

1. **Role scaffolding.** Edge has its own chat template and role tokens. We
   mirror the CONTENT and its ORDER, not the teacher's turn markup — hence
   segments rather than one pre-rendered string.
2. **How many placeholder tokens one image costs.** That is a property of Edge's
   SigLIP2 tower + processor, not of the teacher, and it varies per image with
   the grid the processor picks. `assemble` takes an `image_tokens(i)` callable
   — the caller processes the frames first and answers from the real
   `image_grid_thw`, so nothing here has to guess.

Ego motion rides in the 48 reserved `<|traj_history|>` slots, exactly as in the
teacher — and it is DISCRETE, not continuous. `model.fuse_traj_tokens` sounds
like an embedding fusion but is not: it tokenizes the ego history with a second
tokenizer (`DeltaTrajectoryTokenizer`, 1000 bins) and REPLACES the placeholder
ids with the resulting bin ids (`base_model.py::tokenize_history_trajectory` ->
`replace_pad_token`). So the student needs no projection module at all — it
appends the history bins to its vocabulary alongside the future ones and fills
the slots with ids. (D-029; this corrects D-028, which assumed a learned
continuous encoder was required.)
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Literal

# --- verbatim from the teacher (helper.py / components.py) --------------------

SYSTEM_PROMPT = "You are a driving assistant that generates safe and accurate actions."

# components.construct_user_prompt with components_prompt = ["cot", "traj_future"],
# which is what the released create_message hardcodes.
INSTRUCTION = (
    "output the chain-of-thought reasoning of the driving process, "
    "then output the future trajectory."
)

# helper.py: num_traj_token = 48. token_utils: 128 future bins (D-014).
N_HISTORY_SLOTS = 48
N_FUTURE_TOKENS = 128

# alpamayo.common.constants.CAMERA_NAMES_TO_{INDICES,DISPLAY_NAMES}. The index is
# what orders the cameras in the prompt (construct_image asserts ascending) and
# the display name is what the model was trained to read.
CAMERA_INDEX = {
    "camera_cross_left_120fov": 0,
    "camera_front_wide_120fov": 1,
    "camera_cross_right_120fov": 2,
    "camera_rear_left_70fov": 3,
    "camera_rear_tele_30fov": 4,
    "camera_rear_right_70fov": 5,
    "camera_front_tele_30fov": 6,
}
CAMERA_DISPLAY_NAME = {
    "camera_cross_left_120fov": "Front left camera",
    "camera_cross_right_120fov": "Front right camera",
    "camera_front_wide_120fov": "Front camera",
    "camera_front_tele_30fov": "Front telephoto camera",
    "camera_rear_left_70fov": "Rear left camera",
    "camera_rear_tele_30fov": "Rear camera",
    "camera_rear_right_70fov": "Rear right camera",
}

# Appended to the student vocabulary alongside the 3000 trajectory bins (D-014).
# The student's tokenizer knows none of these; they are structural, and the
# trajectory losses key off the span they delimit.
SPECIAL_TOKENS = [
    "<|traj_history_start|>", "<|traj_history|>", "<|traj_history_end|>",
    "<|route_start|>", "<|route_end|>",
    "<|cot_start|>", "<|cot_end|>",
    "<|traj_future_start|>", "<|traj_future_end|>",
]

SegmentKind = Literal["text", "image", "slots", "bins"]


@dataclasses.dataclass(frozen=True)
class Segment:
    """One piece of the context. `kind` decides how `assemble` expands it.

    text   literal text -> tokenizer
    image  one camera frame -> N placeholder ids, N from the processor
    slots  a special token repeated `count` times (ego-history slots)
    bins   `count` trajectory-bin ids, region-relative (D-014) or None in
           generation mode
    """
    kind: SegmentKind
    text: str = ""
    count: int = 0
    values: tuple[int, ...] | None = None


def user_segments(camera_names: list[str], n_frames: int,
                  nav_text: str | None = None,
                  hist_bins: list[int] | None = None) -> list[Segment]:
    """Cameras -> ego-history slots -> [route] -> instruction.

    `camera_names` are config names; they are sorted by camera INDEX here
    because the teacher's `construct_image` asserts ascending order and the
    model learned that arrangement.
    """
    unknown = [c for c in camera_names if c not in CAMERA_INDEX]
    if unknown:
        raise ValueError(f"unknown camera(s) {unknown}; expected {sorted(CAMERA_INDEX)}")
    segs: list[Segment] = []
    for name in sorted(camera_names, key=lambda c: CAMERA_INDEX[c]):
        # include_camera_ids and include_frame_nums are both true in the A1.5
        # checkpoint config (D-019), so both labels are part of the format.
        segs.append(Segment("text", text=f"{CAMERA_DISPLAY_NAME[name]}: "))
        for f in range(n_frames):
            segs.append(Segment("text", text=f"frame {f} "))
            segs.append(Segment("image"))
    segs.append(Segment("text", text="<|traj_history_start|>"))
    # Placeholder ids when `hist_bins` is absent (rendering, or before the ego
    # history is tokenized); the real bins when it is present. The teacher does
    # the same thing in two steps — emit placeholders, then replace_pad_token.
    segs.append(Segment("slots", text="<|traj_history|>", count=N_HISTORY_SLOTS,
                        values=tuple(hist_bins) if hist_bins is not None else None))
    segs.append(Segment("text", text="<|traj_history_end|>"))
    if nav_text:
        segs.append(Segment("text", text=f"<|route_start|>{nav_text}<|route_end|>"))
    segs.append(Segment("text", text=INSTRUCTION))
    return segs


def assistant_segments(coc_text: str | None = None,
                       traj_bins: list[int] | None = None,
                       n_future: int = N_FUTURE_TOKENS,
                       for_generation: bool = False) -> list[Segment]:
    """The target side. Pass nothing for generation mode (opens `<|cot_start|>`
    and stops, exactly like the released `create_message`).

    `for_generation` with a `coc_text` is the third case, and it is the one the
    stage-1 gate needs: the CoC is teacher-forced but the trajectory is not, so
    the sequence STOPS at `<|traj_future_start|>` and the model decodes from
    there. Without it the prefill would end on `<|traj_future_end|>` (the `bins`
    segment is skipped when `values is None`, the closing token is not), and the
    student would be asked to continue a trajectory that has already been
    closed.
    """
    segs = [Segment("text", text="<|cot_start|>")]
    if coc_text is None:
        return segs
    segs.append(Segment("text", text=coc_text))
    segs.append(Segment("text", text="<|cot_end|>"))
    segs.append(Segment("text", text="<|traj_future_start|>"))
    if for_generation:
        return segs
    segs.append(Segment("bins", count=n_future,
                        values=tuple(traj_bins) if traj_bins is not None else None))
    segs.append(Segment("text", text="<|traj_future_end|>"))
    return segs


@dataclasses.dataclass
class Assembled:
    input_ids: list[int]
    history_span: tuple[int, int]          # [start, end) — where ego motion is written
    image_spans: list[tuple[int, int]]     # one per frame, in prompt order
    coc_span: tuple[int, int] | None       # text between cot_start/cot_end
    traj_span: tuple[int, int] | None      # the 128 bin positions
    n_prompt: int                          # ids before the assistant turn


def assemble(
    segments: list[Segment],
    encode: Callable[[str], list[int]],
    special_id: Callable[[str], int],
    bin_id: Callable[[int], int],
    image_tokens: Callable[[int], int],
    image_token_id: int,
    hist_bin_id: Callable[[int], int] | None = None,
    n_prompt: int | None = None,
) -> Assembled:
    """Segments -> ids + the spans the trainer needs.

    Everything model-specific arrives as a callable, so this is pure and can be
    tested without a tokenizer, a processor, or a checkpoint. `n_prompt` marks
    the user/assistant boundary; pass the length after the user segments.
    """
    ids: list[int] = []
    history_span = (0, 0)
    image_spans: list[tuple[int, int]] = []
    coc_span = traj_span = None
    pending_coc_start: int | None = None

    for seg in segments:
        if seg.kind == "text":
            start = len(ids)
            ids.extend(encode(seg.text))
            if seg.text == "<|cot_start|>":
                pending_coc_start = len(ids)
            elif seg.text == "<|cot_end|>" and pending_coc_start is not None:
                coc_span = (pending_coc_start, start)
        elif seg.kind == "slots":
            start = len(ids)
            if seg.values is None:
                ids.extend([special_id(seg.text)] * seg.count)
            else:
                if len(seg.values) != seg.count:
                    raise ValueError(
                        f"expected {seg.count} history bins, got {len(seg.values)}")
                if hist_bin_id is None:
                    raise ValueError("hist_bin_id is required when history bins are given")
                ids.extend(hist_bin_id(b) for b in seg.values)
            history_span = (start, len(ids))
        elif seg.kind == "image":
            start = len(ids)
            ids.extend([image_token_id] * image_tokens(len(image_spans)))
            image_spans.append((start, len(ids)))
        elif seg.kind == "bins":
            start = len(ids)
            if seg.values is None:
                continue                      # generation mode: nothing to force
            if len(seg.values) != seg.count:
                raise ValueError(
                    f"expected {seg.count} trajectory bins, got {len(seg.values)}")
            ids.extend(bin_id(b) for b in seg.values)
            traj_span = (start, len(ids))
        else:
            raise ValueError(f"unknown segment kind {seg.kind!r}")

    return Assembled(
        input_ids=ids, history_span=history_span, image_spans=image_spans,
        coc_span=coc_span, traj_span=traj_span,
        n_prompt=len(ids) if n_prompt is None else n_prompt,
    )


def render(segments: list[Segment], image_token: str = "<image>") -> str:
    """Human-readable rendering — for eyeballing the context in a log or a test."""
    out = []
    for seg in segments:
        if seg.kind == "text":
            out.append(seg.text)
        elif seg.kind == "slots":
            out.append(seg.text * seg.count if seg.values is None
                       else f"<hist>x{seg.count}")
        elif seg.kind == "image":
            out.append(image_token)
        elif seg.kind == "bins":
            out.append(f"<bin>x{seg.count}")
    return "".join(out)
