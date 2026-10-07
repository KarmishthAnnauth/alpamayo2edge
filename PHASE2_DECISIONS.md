# Phase 2 decisions — the flow head on Bench2Drive

Record of every design decision taken for phase 2 (flow-matching action head),
2026-09-28/29. Companion to `DECISIONS.md` (the project-wide append-only log,
D-001..D-059) and `PHASE1_RUNBOOK.md`. Numbered P2-xx in the order they were
made. Status tags: DECIDED (user), MEASURED (a number on disk), IMPLEMENTED
(code merged and smoked).

Where things stood on 2026-09-28 morning: the student's flow head had never
produced a trajectory (`_gen_pathway_forward` raised NotImplementedError),
stage 2 was scaffolded as teacher-flow distillation on PhysicalAI-AV, Bench2Drive
was 1000 tarballs under `/bulk/datasets/bench2drive`, and no loader existed.

---

## P2-01 [DECIDED] Init checkpoint = RL run 7 step 25, not SFT run 7c

The flow head conditions on the CoC of `runs/stage1_rl/rl-run-7-coc-grounded/step-0025`
(adapters on `runs/stage1/run-351/best`, D-058), folded into the AR tower before
stage 2 starts (`train_stage2.load_student`: inject, `load_adapters`, `merge_lora_`).
7c (job 336) kept epoch 3, the worst epoch on the driver table (D-046/D-047).
Caveat from D-059: step 25 is a lucky sample of its recipe, not what the recipe
reliably yields; it is still the best CoC we own.

## P2-02 [DECIDED] Stage 2 trains on Bench2Drive with `gt_flow`, not the PhysicalAI teacher-flow cache

Data = the CARLA expert's trajectories, loss = flow matching on the expert
action (the objective Alpamayo 1.5's own stage 2 uses). There is no teacher
velocity cache for CARLA data and the user chose not to build one. The
PhysicalAI arm (`stage2.dataset: physicalai`, `supervision: teacher_flow`, D-007)
stays in the trainer as the comparison arm.

## P2-03 [DECIDED] Full fine-tune of the gen tower, fp32 master weights; LoRA r16 is the comparison arm

`student.lora.stage2.enabled: false`. Rationale: the 96 GB card holds it
(measured 35.8-37.5 GiB at micro-batch 8 x 8 samples), NVIDIA's released action
post-training (DROID/LIBERO on Nano) freezes the reasoner and fully trains the
gen tower with fresh head rows at a higher lr, and the domain-31 rows are new
parameters either way. TRAINING_STRATEGY §2's LoRA choice was made under the
48 GB premise. fp32 promotion of the trainable tensors (`stage2.fp32_master`)
because AdamW over bf16 weights loses 1e-5-scale updates; `checkpoint.save`
casts back to bf16 on disk (7.3 GB per epoch).

Frozen or gradient-masked: AR tower + vision tower (the phase-1 model),
`time_embedder`, `action_modality_embed`, the other 31 embodiment rows of
`action2llm`/`llm2action` (row mask, D-017). Trainable: 1,409.4 M gen-pathway
params + 8.5 M interface tables (row 31 only receives gradient, x8 lr).

## P2-04 [IMPLEMENTED] The flow head is our own dense action-only gen pathway

`src/distill/student/flow_path.py`. cosmos-framework asserts "We do NOT support
action only training!" and its UniPC sampler denoises `[vision | action]` as one
state, so the released path cannot be used. Ours reproduces the sub-module
sequence of `MoTDecoderLayer.forward(gen_only=True)` (unified_mot.py:1161-1319)
over `[N, 64, hidden]`: `input_layernorm_moe_gen` -> q/k/v_proj_moe_gen with
q/k_norm_moe_gen -> RoPE -> attention over [reasoner K/V | own 64 tokens] ->
o_proj_moe_gen -> residual -> post_attention_layernorm_moe_gen -> mlp_moe_gen.
Reasoner K/V per layer are `k_norm_und_for_gen(k_norm(k_proj(h)))` + RoPE and
`v_proj(h)`, i.e. what the framework's own AR inference feeds the generator from
its MemoryState. Action positions continue after the last context token.
Smoked on both cards (Ada LoRA arm 11.4 GiB; Blackwell full FT job 482).

## P2-05 [DECIDED] Context = history in, future trajectory tokens out

The action tokens attend prompt + 48 ego-history tokens + route hint + CoC up to
and including `<|traj_future_start|>`; the token path's future bins and what
follows are masked (`flow_path.context_key_mask`). This is what Alpamayo's expert
does (KV crop at `<|traj_future_start|>` in sft_alpamayo_r1.py:166-168, mask in
alpamayo1_5.py:201-204). Conditioning on the future bins would make the head a
refiner of the token path's 3.8 m ADE (D-034). Resolves the divergence noted at
DECISIONS.md:1112.

## P2-06 [IMPLEMENTED] Samples of one window share its context K/V without copying

The first smoke OOM'd at 33 GiB: an `index_select` of 28 layers of K/V per flow
sample cost ~10 GB per forward. Attention now regroups queries per window,
scores the context once, and uses a block-diagonal mask so a sample only sees
its own 64 action tokens. `owner` must be `arange(B).repeat_interleave(k)`.

## P2-07 [DECIDED] Sampler = the teacher's: 10 Euler steps, t 0 -> 1, `x += dt*u`, init `randn*temperature`

Byte-identical in structure to alpamayo1_5 `flow_matching.py:171-191`. Not the
framework's UniPC (joint with video, 30 steps, shift 10). The SDE mode
(Flow-GRPO ODE->SDE conversion, per-step Gaussian log-probs) is implemented for
DiffGRPO but UNVERIFIED against the paper - see TODO.

## P2-08 [DECIDED] Timestep draws: K = 8 per window, stratified over [0, 0.999)

`stage2.k_samples: 8`, `stage2.t_sampler: stratified` = the labeler's
`TeacherWrapper.stratified_timesteps` rule (0.999 cap keeps the a0 recovery
well-conditioned). `logitnormal` (Edge's own train-time sigma law) is the switch.
`a_t = t a1 + (1-t) a0`, target `a1 - a0`, MSE in fp32 (D-012/D-016 conventions;
`flow_forward` converts to the student's sigma = 1-t, v = -u internally).

## P2-09 [DECIDED] Bench2Drive extracted in full, tarballs deleted, unused modalities repacked

User instruction: delete each tarball once its extraction is verified (file
count vs archive listing, `scripts/01c_b2d_extract.sh`). All 1000 clips done;
one archive had a download-truncated name (`..._Weathe.tar.gz` = `..._Weather2`)
and was finalised by hand. Then [MEASURED] the /bulk USER quota is on file count
(7M hard limit, hit at 940 clips); user chose to repack depth / semantic /
instance / lidar / radar into one `extras.tar` per clip
(`scripts/01c_b2d_repack.sh`, verified, reversible with `tar xf`) rather than
ask for a raise. Inodes 7.0M -> 2.4M; bytes unchanged (542 GB of 750 GB).

## P2-10 [DECIDED] Window definition and camera mapping

Anchor t0, 16 history + 64 future steps at 10 Hz, stride 20 frames, frames at
t0-3..t0. Cameras onto the four PhysicalAI slots: cross_left <- rgb_front_left,
front_wide <- rgb_front, cross_right <- rgb_front_right, front_tele <- centre
crop of rgb_front with 30 deg horizontal FOV (x in [494,1106], 612x344, 16:9),
all resized to 360x640. No back camera (phase 1 used none). FOV mismatch
(CARLA 70 deg vs PhysicalAI 120 deg on the three wide slots) is accepted and
recorded in `raw_cameras`.

## P2-11 [MEASURED] Ego pose from the bbox `world2ego`, y flipped, 1.25 m rear-axle shift

Top-level x/y in the anno are noisy; the ego bbox pose is smooth. CARLA is
left-handed (y right) -> flip to the +y-left ego frame (D-038). CARLA's actor
origin is mid-body and the unicycle assumes velocity along heading (true at the
rear axle), so `REAR_AXLE_OFFSET_M = 1.25` (fitted on 8 turning clips: |yaw -
travel| 3.0 deg -> 0.41 deg; round-trip error on turning windows 0.535 ->
0.196 m median). Residual `action_to_traj(traj_to_action(x))` floor is the
action space's own (p50 0.03 m, p90 0.50, from low-speed curvature clamps and
tyre slip), stored per window as `action_floor_m`.

## P2-12 [DECIDED] Route hint from Bench2Drive commands, "horizon" rule

`route_source: command`: first non-LANEFOLLOW `command_near` within the 64-step
horizon, else near at t0 (the near target is always 1-7 m ahead, so the
"near unless < 5 m" rule degenerates). Mapped onto phase 1's three hint strings;
lane changes read "Continue straight" (no class for them). Deployment-faithful:
Bench2Drive supplies commands, the kinematic hint (phase 1's) would leak the
future.

## P2-13 [DECIDED] Splits by town + route, ~10% of routes, stratified by scenario

894 train / 105 val clips = 8,040 / 863 windows; all 43 scenarios have >= 1 val
route; weathers of a route stay together. Same towns on both sides - measures
cross-route, not cross-map, generalisation. One clip (67 frames) is too short
for a window and is dropped.

## P2-14 [DECIDED] CoC generated offline by the init and teacher-forced into the context

`scripts/05j_b2d_coc.py generate` (Blackwell job 483, 0.7 s/window, 8,903 rows,
all terminated) -> `b2d_cache/coc_rl7s25.jsonl` = `stage2.coc_cache`. The flow
head trains on the reasoning it will be handed at deployment. `null` = no CoC in
the context (ablation).

## P2-15 [MEASURED] The CoC on CARLA renders: consistency transfers, braking anticipation does not

`05j score`, all 8,903 windows [val 863] vs the same checkpoint on PhysicalAI val
(D-058): GT-consistent 0.765 [0.735] (PAI 0.780); false-clear 0.149 [0.175]
(PAI 0.288); braked -> says slow/stop 0.215 [0.183] (PAI 0.385); expert STOPPED
0.231 [0.205] (PAI 0.750); hard brakes 0.045 n=110 (PAI 0.379); NUDGE 3.6%;
maneuver mix STOP 38% / FOLLOW 34% - STOP is said where the ego already stands.
NOT a student-vs-teacher comparison: the teacher was never run on CARLA (user:
leave that). CARLA's stops are scripted events often invisible 6.4 s ahead.

## P2-16 [DECIDED] Keep the teacher's action normalisation for SFT run 1

[MEASURED] The CARLA expert's accel std is 3.02 m/s^2 vs the teacher's 0.681, so
in the teacher's units the train actions have std 4.4, 14.6% of accel entries
beyond 5 sigma, max 50 sigma; expected MSE(a1 - a0) per entry 11.8 instead of 2.
Run 1 launched as is (user: "Kick off"). Re-normalising with Bench2Drive's own
statistics for the domain-31 head is the TODO for run 2.

## P2-17 [DECIDED] Gate = 160 val windows, seeded subset, k = 6, 10 Euler steps, minADE vs expert; every epoch saved; `best` = lowest gate minADE

Replaces the dead `05_eval.py`. Shared code `src/distill/eval/flow_minade.py`;
standalone `scripts/05k_flow_minade.py` for full val, per scenario, braking vs
not, next to the constant-velocity baseline (zero normalised action, what an
untrained head emits). The subset is a seeded permutation, not an alphabetical
prefix (the smoke's prefix was one scenario). Still an open-loop proxy; the
deliverable is closed-loop Driving Score.

## P2-18 [DECIDED] SFT run 1 = job 485, launched 2026-09-28 23:59

`sbatch scripts/04_train_stage2.sh`, run name `sft-run-1-b2d-fullft`: 8 epochs
x 252 steps, micro-batch 8 windows x 8 samples, grad accum 4, lr 5e-5 (x8 on the
interface rows), AdamW (0.9, 0.95), wd 0, cosine with 3% warmup, clip 1.0, bf16
autocast + fp32 master, per-layer checkpointing, 37.5 GiB peak, ~55 min/epoch.
[MEASURED] epoch-0 gate: minADE_6 3.20 m vs constant-velocity 5.99 (braked
windows 3.62 vs 6.59, others 0.86 vs 2.59); loss 5.86 at step 20 -> 1.24 at
step 200.

[MEASURED 2026-09-29] Run 1 DONE 07:19 (7 h 20 min). Gate minADE_6 per epoch 0..7: 3.20, 2.99,
2.59, 2.64, 2.22, 2.11, 1.99, 2.02 -> `best` = epoch 6. Full val (job 486, 863 windows,
`runs/flow-minade-sft1-best-valfull.json`): minADE_6 2.18 m vs constant-velocity 5.99;
meanADE 4.24, minFDE 4.53, p90 5.16; braking windows (687) 2.44 vs 6.77, others (176) 1.13
vs 2.98; 106/863 windows worse than constant velocity. By initial speed: standing (v0 < 0.5,
n=230) 1.28 vs 2.81; 0.5-5 m/s (147) 3.16 vs 8.84; 5-10 m/s (359) 2.54 vs 7.83; > 10 m/s (15)
4.13 vs 8.93. Worst scenarios: InterurbanActorFlow 4.75 (n=4), ControlLoss 4.25,
YieldToEmergencyVehicle 4.02, junction left turns ~3.4; best: green-light turns 0.65,
ConstructionObstacleTwoWays 1.09, CrossingBicycleFlow 1.10. Judge further checkpoints with
`scripts/05k_flow_minade.py --ckpt runs/stage2/<run>/epochN` on full val.

## P2-19 [DECIDED] After SFT: DiffGRPO, open-loop replay reward first, closed-loop CARLA second

Order agreed 2026-09-29 (see TODO). Not started. GPU etiquette unchanged: the
Ada behind its guard without asking (it was occupied by another user's three
CARLA servers on 2026-09-28 evening, hence the Blackwell for the CoC cache);
every Blackwell `sbatch`/`scancel` announced first.

---

## TODO

### T1. Re-normalise the action space with Bench2Drive statistics (SFT run 2) - DONE, run 2 = job 487
Launched 2026-09-29 08:11 (`sft-run-2-b2d-fullft-b2dnorm`, W&B run of the same name).
Stats `configs/b2d_action_stats.json` (train split, physical units: accel mean 0.109 / std
3.025 m/s^2, curvature mean 0.0024 / std 0.0358 1/m) -> cache
`/bulk/users/vla/alpamayo2edge/b2d_cache_b2dnorm` (builder `--action-stats`, constants stored in
`<cache>/action_space.json`, read back by `bench2drive.load_cache_action_space` in the evaluator).
Verified: identical splits, `gt_future_xyz` identical, `gt_traj` an exact affine remap (max err
1e-6), new-unit std 1.01 / 0.98, expected MSE(a1 - a0) 1.99. Same CoC cache, same recipe as
run 1 otherwise. Compare on the same 160-window gate and full val; still one seed each.
Original plan:
Compute accel/curvature mean/std over the train windows (measured 2026-09-28:
accel std 3.02 m/s^2, curvature std 0.0358 1/m vs the teacher's 0.681 /
0.0261) and build the cache with those constants for the domain-31 head
(`bench2drive.load_action_space` kwargs, ~3 min rebuild; `05k`/gate use the
same space so metres are unchanged). Nothing physical changes: the domain rows
are our own affine interface (D-017). Expected effect: the loss stops being
owned by emergency stops, and the N(0, I) prior matches the data scale at
sampling. Keep `gt_traj_token_ids` in the teacher's bins (they are not in the
flow context). Compare run 2 to run 1 on the same gate; two seeds before any
claim (D-059).

### T2. DiffGRPO with a replay-based (non-reactive) reward
Reward for a sampled trajectory rolled out kinematically against the logged
scene: collision with any actor's logged future box (`bounding_boxes` per
frame), off-road once `rethinklab/Bench2Drive-Map` is downloaded, progress along
the route, NVIDIA's comfort bounds (alpamayo1_x_rl `comfort_reward.py`), and
agreement with the expert's brake flag. Reference: ReCogDrive's simulator-
assisted GRPO on a diffusion planner. Before the first RL step: VERIFY the SDE
sampler (`flow_path.sample_actions(sde_noise=a)`, written from recollection of
Flow-GRPO) against the paper's equations - marginal preservation, the first-step
clamp - and add an offline test. Loop mechanics from `train_grpo_coc.py` (group
sampling, advantages, clipping); keep the flow-matching loss on expert windows
as the imitation term. Audit every reward term for a hedge that cannot lose
(phase-1.5 lesson).

### T3. Closed-loop RL in CARLA on scenario snippets
Agent wrapper for the vendored harness (`~/projects/TriTrack/TriTrack/Bench2Drive`,
CARLA 0.9.15, template `master_thesis/team_code/qwen3vl_b2d_agent.py`); slow-fast
split (refresh CoC + context K/V every ~2 s, flow head every control step);
group = G episodes of the same route with different sampler noise, return = the
episode's driving-score components, gradient through the per-step SDE
log-probs; optional dense signal from a privileged planner (CaRL) in the same
scenario. Budget check first: ~1-2 plans/s per policy and a shared box.

### T4. W&B on every stage-2 run (IMPLEMENTED 2026-09-29, first used by run 2)
`train_stage2._WandB`: one run per job named `<run_name>-job<id>` in project
`alpamayo2edge`, `wandb.alert` on start / finish / failure (W&B forwards these to
email or Slack per the account's alert settings), per-step `loss/*` and `lr`,
per-epoch `gate/*`. Config `stage2.wandb`, `stage2.wandb_project`. Job 485 (run 1)
predates it; watch that one through `logs/a2e-stage2-485.out`.

### T5. Housekeeping
- `05_eval.py` is dead; delete or point it at `05k_flow_minade.py`.
- Teacher CoC on the same Bench2Drive windows (a student-vs-teacher table on
  CARLA) - explicitly parked by the user on 2026-09-29.
- DECISIONS.md D-060 entry for the full-FT / Bench2Drive decision - write when
  asked.
