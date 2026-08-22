# Training Strategy — adaptation, not compression

**Status as of 2026-08-19.** Decided but **not yet implemented** — see §6.
Companion to `DECISIONS.md` (the append-only log; this file is the reasoning
behind D-024/D-025 and the framing that motivates them).

---

## 1. What this project actually is

Three different things get called "distillation". They have different SOTA
practice, and we had been importing habits from the wrong one.

| | Setting | Standard practice |
|---|---|---|
| **1. Compression** | Student is a smaller version of the teacher; goal is retaining teacher performance with fewer params. DistilBERT/TinyBERT/MiniLM, Minitron (prune-then-distill), sequence-level KD (Kim & Rush). | **Full fine-tuning**, because the student is effectively being re-pretrained on billions of tokens near the teacher's own distribution. Nothing to forget: the student starts from scratch or from a pruned copy of the teacher. |
| **2. Capability transfer into a strong generalist** | A capable pretrained model acquires a specialist's skill. Instruction distillation (Alpaca/Vicuna-style SFT on teacher generations). | **Parameter-efficient adaptation**, small curated data, teacher outputs as the supervision signal rather than as a compression target, plus explicit retention checks. The student's priors are the asset. |
| **3. Step distillation** | Consistency / progressive / distribution-matching distillation for diffusion and flow models. | Compresses *sampling steps*, not capacity. Unrelated mechanism (already distinguished in D-011). |

**We are in bucket 2.** Cosmos 3 Edge was chosen *because* it already has strong
physical grounding; the teacher supplies AV-specific reasoning and trajectory
behaviour that Edge lacks. Three consequences:

- **The parameter ratio stops mattering.** The A2 Super -> A1.5 swap dropped the
  reasoner ratio from 32B->4B to 8B->4B (D-018). Under a capability-transfer
  framing that is close to irrelevant — nobody asks the parameter ratio in
  instruction distillation; they ask whether the capability transferred and
  whether the base model survived.
- **LoRA is the standard method here, not a compromise.** Full FT in bucket 1 is
  justified by data volume we do not have: ~10k windows is five to six orders of
  magnitude below a distillation pretraining corpus.
- **The headline result is a pair of numbers**, task gain *and* retention,
  because the claim is "added a skill without breaking the model".

**Thesis framing:** prefer *domain-adaptive capability transfer* over
"distillation into" in the title/abstract. "Distillation" primes readers for
bucket 1 and invites compression questions we do not want to answer.

---

## 2. Decision: LoRA for both stages (D-024)

Previously Stage 1 full-FT'd the 1.94B AR tower and Stage 2 full-FT'd the 1.94B
gen tower. That was never a considered anti-forgetting choice — D-015's freeze
schedule was about *tower separation* (which parameters belong to which stage),
and "the other tower is fully trainable" came along with it, inherited from the
96GB premise where full FT was simply affordable.

**Where the risk actually was.** The physical prior lives mostly in the **gen
tower** — the video/world-model half. Stage 1 leaves it frozen, so Stage 1 mainly
risked the AR tower's general VLM breadth (VQA, open-vocabulary grounding), which
is partly irrelevant to deployment but *not* irrelevant to the long tail — the
whole point of curating cut-ins and construction zones is generalization to
scenes we did not train on. **Stage 2 was the real problem**: it unfroze all
1.94B of the world-model tower to learn a 64x2 velocity field from ~10k windows.

**The framework already disagreed with us.** `cosmos-framework` ships a
first-class LoRA path (`cosmos_framework/utils/generator/lora.py`:
`inject_lora_pre_fsdp`, `init_lora_weights_post_materialization`, optimizer
filtering via `keys_to_select=["lora_"]`), and `Cosmos3-Edge.yaml` defaults to:

```yaml
lora_rank: 16
lora_alpha: 32
lora_target_modules: q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen
```

Those targets are exactly the **gen tower's** attention projections. NVIDIA's
intended adaptation path for the generation tower is LoRA, not full FT.

### What is low-rank vs full-rank

The distinction that matters is **new interface** (full-rank) vs **existing
pretrained weights** (low-rank):

- **Low-rank (LoRA):** Stage 1 = AR-tower `q/k/v/o_proj`, rank 32-64 (more
  behaviour to absorb). Stage 2 = gen tower, rank 16 per the shipped default.
  MLP targets are the escalation if attention-only underfits; note leaf names
  `up_proj`/`down_proj` are shared across towers, so they must be path-qualified
  (`mlp_moe_gen.up_proj`) — the injector's docstring covers this.
- **Full-rank, always:** the 3000 new trajectory-vocab rows and the new
  action-embodiment rows in `DomainAwareLinear` (D-017). These are new,
  randomly-initialized parameters; LoRA on them is meaningless. ~12M params.
- **Keep** `new_token_lr_mult: 8.0`.

### Gradient masking (do not skip this)

`inject_lora_pre_fsdp` freezes every non-LoRA parameter, so the new params above
must be re-enabled explicitly — and then they need masking, or forgetting leaks
straight back in through the tensors LoRA does not cover:

- **Embedding / lm_head:** train only rows `>= old_vocab_size`. Zero the
  gradient for the 131,072 pretrained rows.
- **`action2llm` / `llm2action`:** train only the row for our
  `action_domain_id`. Training the whole tensor would damage the other 31
  embodiments' action heads.

Known cost of the row-mask approach: AdamW still allocates optimizer state for
the *whole* embedding tensor (~7.7GB). Acceptable now that LoRA freed ~35GB. The
memory-optimal alternative (separate trainable embedding + concatenated logits)
was rejected as too fragile around `generate()`.

### The one real cost

LoRA may underfit, and then a failed Stage-1 coarse-minADE gate is ambiguous:
"LoRA too weak" vs "the distillation signal is weak". **Mitigation:** run full-FT
as an ablation on the **500-clip increment only**. Cheap, and it buys back the
diagnostic.

---

## 3. Memory budget — RTX 6000 Ada, 48 GB

**Teacher labeling fits comfortably (~30 GB).** 22.2GB weights + ~0.5GB prompt KV
(4 cameras x 4 frames ~= 3.3k tokens) x4 for `expert_batch` + vision/activations.
`expert_batch` could go to 8. The flow-target cache (D-007) is what keeps teacher
and student from ever coexisting in memory.

**Student full FT did not fit.** Edge is `Nemotron-2B-Dense-VL`: hidden 2048,
28 layers, 8 KV heads, intermediate 9216, vocab 131072, untied embeddings.

| | params |
|---|---|
| AR tower (28 layers) | 1.94 B |
| Gen tower (`_moe_gen` duplicate of attn+MLP) | 1.94 B |
| Embeddings + lm_head (untied) | 0.54 B |
| **Total** | **4.41 B** (matches the `cosmos3_ga_4bm2b` experiment name) |

Stage 1 full FT = 2.48B trainable: 8.8GB weights + 5.0GB grads + 19.8GB AdamW
fp32 states + 9.9GB fp32 master = **43.5 GB before a single activation**. Does
not fit at any batch size. (NVIDIA's own Edge recipe runs
`data_parallel_shard_degree: 8` with `fsdp_master_dtype: float32` and EMA on —
it is an 8-GPU config.)

**With LoRA**, Stage 1 trainable drops from 2.48B to roughly **25M**: ~9GB of
weights plus activations. `micro_batch` can go **up**, not down.

**Superseded:** the earlier 48GB mitigations (8-bit AdamW, frozen-embedding
trick, `micro_batch: 1`) are no longer needed. **EMA stays off.**

---

## 4. Retention — measure it, do not assume it

The plan has no retention metric anywhere, so as written it could not detect
forgetting if it happened. Add to `eval/`: score the base Edge checkpoint and
each trained checkpoint on something the AV data does not cover — a handful of
general video/image QA items, plus the gen tower's own denoising loss on generic
clips. Two numbers, before and after.

Under the §1 framing this is not defensive hygiene, it is **half the result**.

---

## 5. Experiment design consequences

- **The load-bearing comparison is teacher-supervised vs GT-only, at matched data
  and matched compute.** The config already has the pieces (`gt_ce`, `fm_gt`, the
  `gt_flow` ablation switch) — make it the headline, not an ablation. Without it
  a reviewer asks: "you fine-tuned on 5k AV clips; how do we know the teacher did
  anything?"
- **Why the teacher is needed at all** (state this explicitly in the thesis): the
  dataset has no chain-of-causation labels — the teacher generates them; a single
  GT trajectory cannot express multimodality — the teacher's 3000-bin
  distribution can; one GT sample per window is far sparser supervision than a
  velocity field queried at K noise levels.
- **Off-policy by construction.** Modern generative KD has moved toward on-policy
  distillation (MiniLLM; GKD, Agarwal et al.) because offline KD trains the
  student on states the *teacher* visits while evaluating it on states the
  *student* visits. Our cached-target design is inherently off-policy — that is
  the price of single-GPU feasibility, and `scheduled_sampling_start_frac: 0.5`
  is a partial mitigation. **Do not change it; name the tradeoff in the thesis**
  rather than letting an examiner find it.

*Citations above are from recollection (assistant knowledge cutoff ~May 2026) and
should be verified before they go in the thesis.*

---

## 6. Implementation status — APPLIED 2026-08-22

All five items below are done; see **D-026** for what was built and for four
latent bugs the wiring turned up. Status header kept in place because the
sequencing that follows still matters.

1. ~~Rewrite the param-group methods around the real LoRA API.~~ Done —
   `student/lora.py` + `EdgeStudent.param_groups_stage{1,2}`. Injection is
   scoped to `language_model.model.layers`, NOT the causal LM: the framework
   matches plain targets by leaf name, and the SigLIP2 vision tower shares
   `q_proj`/`k_proj`/`v_proj` with the AR tower.
2. ~~Add `merge_lora_()` before the Stage-1 save.~~ Done, but as a
   **non-destructive** `checkpoint.save` — merging in place mid-training would
   end the run, and Stage 1 saves every time the gate improves. `merge_lora_()`
   also exists, for export.
3. ~~Gradient-mask the shared row-indexed tensors.~~ Done, and pinned by tests.
4. ~~`configs/default.yaml`.~~ Done. `micro_batch` doubled with `grad_accum`
   halved (effective batch stays 32). No EMA anywhere — nothing to turn off.
5. ~~Retention eval.~~ `eval/retention.py` + 24 non-AV probes + a
   `baseline`/`score` script. **Partially:** the reasoner half is complete and
   runnable; the gen tower's denoising loss on generic clips is NOT wired (it
   needs a real framework data batch for `training_step`, which is not
   derivable offline with confidence). Relative weight drift per tower covers
   the gen tower in the meantime, and is exact.

Two corrections to what this section previously claimed:

- **`OmniMoTModel.add_lora` does exist** (`omni_mot_model.py:5537`). The old
  `_attach_lora` would still have failed, because it used a signature that
  matches neither the method nor the free function — right conclusion, wrong
  reason.
- The `to_empty(device=...)` note was right, with one addition: it must be
  applied to the `lora_A`/`lora_B` **submodules**. Calling it on the wrapper
  discards the pretrained base weight the framework's design goes out of its
  way to preserve.

New blocker, not hardware-related: **D-027** — `ar_forward` and
`generate_traj_tokens` are written against the HF causal-LM interface, but that
class's `forward` takes a `SequencePack`. The reasoner API is
`model.reasoner_forward` / `generate_reasoner_text`. Stage 1 cannot run until
that is ported, and the port overlaps the chat-template question Stage-1 input
assembly is already waiting on.

### Then, on the GPU box (RTX 6000 Ada 48GB)

Environment + HF auth -> `scripts/00_verify.py` (config-only) -> **the D-022
Phase-B probe**, which is the gate: if Alpamayo 1.5 does not emit usable discrete
trajectory tokens, Stage 1 loses its primary target and the plan changes shape.
Do not start Stage-1 input assembly or any labeling run before that answers.
