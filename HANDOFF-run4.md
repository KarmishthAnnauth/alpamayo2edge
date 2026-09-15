# Handoff — implement GRPO run 4 (phase 1.5), 2026-09-15

Read this, then `DECISIONS.md` D-036..D-040 (append-only; everything below is justified
there with numbers). The previous session ran three RL runs on the student's
chain-of-causation (CoC) and ended on a measurement that changes the design.

## State at handover

**Jobs.** Job 320 = RL run 3 (4 cameras, trajectory reward) was at step ~150/200 on the
Blackwell, ~4 h from done (ETA ~13:00 on 2026-09-15). Log `logs/a2e-grpo-coc-320.out`;
checkpoints `/data/vla/alpamayo2edge/runs/stage1_rl/rl-run-3-traj-4cam/` (`best/` merged =
step 125, `step-NNNN/adapters.pt` at every eval). **Job 321 (user `vqa`, `internvl4b-bf16-*`, partition
`debug`) is queued behind it** and will take the Blackwell when 320 exits - so the Blackwell is
NOT free after run 3. Check `squeue -o "%i %u %j %t"` before planning run 4's launch; queue
behind `vqa` with `sbatch` (SLURM orders it) and tell the user.

**GPU rules (user's, standing).** The Ada (RTX 6000, 47 GB) is shared and unarbitrated -
roberto's two `watch_task.py` demo servers (~1.3 GB, weeks of uptime) live there; use it
freely for inference/smokes via the launchers' guard (refuse if others hold > 4 GB; cap with
`A2E_MEM_FRAC`), never kill anything on it. The Blackwell (95 GB) is SLURM-managed
(`sbatch`, partition `main`); **tell the user before using it**. Launching a run the user
asked for is fine; restarting one because of your own bug - say so first.

**Checkpoints that matter.**
- SFT, 1 camera: `runs/stage1/run-314/best` (r48 attention LoRA, epoch 1; D-036).
- SFT, 4 cameras: `runs/stage1/run-253/best` (r96+MLP, epoch 3). Better CoC baseline on
  every teacher-free metric (D-038, D-040 tables); the RL injects fresh r48 adapters on any
  merged checkpoint, so either is a valid init.
- RL run 1 `rl-run-1/best` (step 400, teacher-match reward - superseded, D-038).
- RL run 2 `rl-run-2-traj/step-0200/` adapters on run-314 (1 cam; superseded - route-hint
  defect, D-039 outcome).
- RL run 3 `rl-run-3-traj-4cam/` (see above; the 4-camera baseline is the useful part).

**Commits.** Everything is committed through `35ba5db` except: `scripts/05d_coc_intervention.py`
(D-040's probe), `DECISIONS.md` D-040, this file, and the `logs/*.out` of jobs 319/320 and
the probes. Commit them first (`git add` those explicitly; do not `git add -A` the logs dir
blindly - it holds every run's log and that is intended, but check sizes).

## What the three runs established (numbers in D-037..D-040)

1. GRPO on the AR tower works on one GPU (~85 s/step at 1 cam G=16, ~270 s at 4 cam).
2. A reward that matches the teacher's text fixes the CoC collapse in ~75 steps and then
   learns the teacher's narration (D-038: the teacher's hazard mentions predict nothing
   about the driver, its turn direction is opposite 17% of the time, it counsels caution the
   driver didn't take). Do not go back to it.
3. A route hint from the GT future must be turn-left / turn-right / straight ONLY (final
   heading > 40 deg). "Change lane" from lateral offset fired on 33% of windows (D-039).
4. **The student's trajectory ignores its CoC (D-040).** Forcing "turn left" vs "turn
   right" moves the decoded heading by 2-6 deg; "stop" vs "accelerate" moves end speed by
   0-1.4 m/s. So an ADE reward trains trajectory tokens only; the CoC gets noise. This is
   the fact run 4 is designed around.

## Run 4 design

Keep: joint rollout (`EdgeStudent.generate_coc_and_traj`), route hint, 4 cameras from
`run-253/best`, G=16, maneuver-weighted prompts, KL 0.03 + CE anchor 0.1, adapters saved
per eval, `select_on` a val metric. Change three things:

### 1. Per-span advantages (the core change)
Today (`src/distill/train_grpo_coc.py`, `mode == "traj"`): one scalar reward per rollout
-> one group-normalised advantage -> applied to both spans. Replace with two rewards per
rollout and two advantages:
- `r_traj` = `-min(ADE, ade_cap)/ade_cap` (+ self-consistency, below) -> advantage
  applied to the **trajectory** span only.
- `r_coc` = `gt_reward.gt_reward(...)` (the GT-grounded text rules: `kin 1.0, dir 0.5,
  hazard 0.5, teacher 0.25`; already implemented, used by `mode: gt`) + self-consistency
  -> advantage applied to the **CoC** span only.
Both group-normalised separately (`group_advantages`), each with its own zero-variance
skip. Failures (unterminated CoC / undecodable trajectory) get `fail` on BOTH.
Where: the rollout loop builds `rewards` and then `adv = group_advantages(rewards)`; make
that two lists (`rew_traj`, `rew_coc`) -> `adv_traj`, `adv_coc`; in the scoring block the
per-span loop `for span, wspan in (("coc", 1.0), ("traj", w_traj))` already separates the
spans - pass `A_coc` to the coc span and `A_traj` to the traj span. `keep` filtering and
the `share` chunk weighting stay as they are.

### 2. Self-consistency term (builds the coupling)
`gt_reward.kinematic_term(student_maneuver, gt_reward.kinematics(student_xyz))` - the same
scorer, pointed at the student's OWN decoded trajectory instead of GT (`student_xyz` is
already computed in `_rollout_traj` for the ADE; return it alongside `ade`). Also
`direction_term` against the plan's own `lateral`. Add with weight `self_consistency`
(start 0.5) to BOTH `r_traj` and `r_coc`: the CoC is rewarded for describing the plan, the
plan for matching the words. Teacher-free, GT-free. Log it as its own component.

### 3. CoC sampled at T=1.0 in rollouts
The 4-camera student is sharp; at T=0.6 its 16 rollouts of a prompt mostly write the same
CoC, so there is nothing for the CoC advantage to choose between. In
`generate_coc_and_traj` (`src/distill/student/edge_wrapper.py`) the `free` sample uses one
`temperature`; add a `coc_temperature` argument used for phase 0 only (trajectory bins stay
at the teacher's 0.6). Plumb `stage1_rl.coc_temperature: 1.0`. Val (`_val_eval`) keeps
0.6 - it is the judged setting.

### Config (`configs/default.yaml`, `stage1_rl`)
`run_name: rl-run-4-perspan`, `reward.mode: perspan`, `reward.self_consistency: 0.5`,
`coc_temperature: 1.0`, `select_on: gt_score` (the CoC metric; ADE is also logged), keep
`eval_windows 150`, `score_chunk 4`, `steps 200`, `prompts_per_step 8`. Document every
change in the config comments - that is the project's convention (see the existing block).

### Tests
`tests/test_grpo_coc_offline.py` and `tests/test_gt_reward_offline.py` cover the pure
parts; add cases for the per-span reward split (a rollout with a good plan and a bad CoC
must get opposite-sign advantages on the two spans) and for self-consistency (a "stop" CoC
with a stopping plan scores +1, with an accelerating plan -1). `python -m pytest tests/ -q`
(132 pass at handover).

### Smoke, then launch
- Smoke on the Ada: `python -m distill.train_grpo_coc --config configs/default.yaml --smoke`
  (G=4, 2 prompts, 2 steps) via a launcher like
  `/tmp/claude-*/scratchpad/smoke_grpo_gt.sh` - or just copy its guard block:
  pin the Ada UUID, refuse if others > 4 GB, `A2E_MEM_FRAC=0.5`, `PYTHONPATH=src:scripts/_ada_shim`.
  At 4 cameras the smoke OOMs below ~28 GB; if the Ada is busy, the Blackwell run's first
  steps are the smoke (each step is ~4.5 min; a failure costs one resubmit).
- Launch: `sbatch scripts/03c_grpo_coc.sh` (after telling the user). Log lands in
  `logs/a2e-grpo-coc-<job>.out`; provenance line shows the commit - commit first.
- Watch: val lines are `step N val: {json}`; a `neg_ade` best is saved to `best/`, adapters
  to `step-NNNN/`. Note `best_acc` is initialised to -inf now (was -1.0, which silently
  never saved when selecting on neg_ade - fixed in `766ba99`).

### Evaluate
- Per-step: `05b_eval_coc.py --ckpt <init> --adapters <run>/step-NNNN --route-hint
  --cameras <4 cams> --split val --n 500` (Ada), or `INIT=... AD=... TAG=... ROUTE=1
  sbatch scripts/05b_sbatch_eval.sh` (Blackwell; add `--cameras` to that script if needed -
  it does not pass it yet).
- Teacher-free metrics from the dump: `gt_reward.gt_metrics` per row (the snippet in this
  session's history; a 15-line script). Compare to run-253 @4cam baseline: GT-consistent
  0.74, GT false-clear 0.21, hazard-ungrounded 0.14, direction 0.18 (150-window val, step 0
  of job 320), and to the teacher's own 0.69 / 0.16 / 0.24 / 0.16.
- Coupling: rerun `05d_coc_intervention.py --ckpt run-253/best --adapters <run-4 step>
  --cameras <4 cams>`. Run 4 succeeds if the "turn left" vs "turn right" heading gap grows
  from ~3 deg and "stop" vs "accelerate" end-speed gap from ~1 m/s - that is the coupling
  being built - AND the CoC mix moves off FOLLOW 53 / TURN ~3.
- Visual: rebuild the viewer (`/tmp/claude-*/scratchpad/coc_viewer/build.py` + template;
  artifact https://claude.ai/code/artifact/0acd3ecd-a5b2-4d2f-a488-c1d3d8e35d09) with the
  new dump - the user judges by looking at frames next to traces.

## When job 320 finishes
Val curve at handover (150-window single-sample ADE): 3.79 / 3.55 / 3.74 / 3.76 / 3.72 /
3.44 / 3.69 at steps 0..150 - i.e. FLAT at ~3.7 m; the 3.44 that `best/` (step 125) was
selected on is a low draw. CoC metrics unchanged at every check (FOLLOW 53-60, TURN <= 6,
direction read 0.12-0.24, GT-consistent 0.71-0.74). Treat run 3 as "no effect" unless the
500-window scoring says otherwise.
Score its best (step 125 or later) on 500 val with `--route-hint --cameras <4 cams>` and
put it in the table; expect ADE improved (~3.3-3.4 m single-sample from 3.79) and the CoC
metrics unchanged - that is the D-040 prediction and the run-4 motivation. Write the D-039
outcome for run 3 in `DECISIONS.md`.

## Pitfalls met this session (don't repeat)
- `pkill -f <pattern>` matches your own Bash tool's shell (exit 144). Use `pgrep`/`ps`
  with a `[p]ython` trick, or kill by PID.
- Background jobs launched via the harness can be killed by its host-memory heuristic
  (it reads `free`, not `available`; the page cache looks like pressure). Launch long
  jobs with `setsid nohup ... & disown` and watch the log file.
- Monitors expire after 30 min regardless of `persistent`; re-arm on the expiry notice.
- Selection on a 99/150-window single-sample check is +/-0.04 acc, +/-0.2-0.3 m ADE.
  Always score candidates on 500 windows before calling anything a result.
- A wait loop that `pgrep`s for `train_grpo_coc` will match the Blackwell job too.
- `05b_eval_coc.py` results JSONs from before the `cameras` field was added lack it.
- The smoke and the real run share `run_name`; the smoke appends to the same `steps.jsonl`
  (a `step 2` row shows up in the curve - ignore or give smokes their own run_name).
