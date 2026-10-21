# Compute platform research: Together.ai and alternatives for DFlash2 runs

Researched October 2026, motivated by disk-space limits on the RunPod B200 big runs.
All prices are on-demand per-GPU-hour unless noted, scraped from provider pages /
computeprices.com in Oct 2026 — treat as ±10% and re-verify before committing money.

## TL;DR

1. **Managed fine-tuning APIs cannot host DFlash2.** Not Together's headline
   fine-tuning product, not Fireworks, not Crusoe serverless FT, not Nebius Token
   Factory. They train a *standard* HF CausalLM with canned SFT/DPO/LoRA recipes.
   Our pipeline needs a custom architecture (tapped-layer drafter + MTP head), a
   pinned custom loss (dpace, γ=7), the patched speculators 0.8.1 trainer, and our
   own vLLM stack (exact deployed W4A16 quant, SERVING_PARAMS sampling). None of
   that is expressible in any managed FT API.
2. **Together has three distinct products and only one is relevant**: their new
   self-serve **GPU Clusters** (Slurm/K8s, SSH login nodes, BYO containers,
   persistent shared volumes) could run our stack as-is. Their **Post-Training**
   custom-loop product looks tempting but still runs *their* runtime with
   catalog-only models (no Qwen3.5 4B/0.8B) — no custom architectures.
3. **The disk problem is fixable in place**: RunPod network volumes go to 4 TB
   ($0.05/GB/mo), attach at pod creation, DC-pinned. Zero workflow change.
4. **Best same-workflow alternative**: Lambda — B200 on-demand $6.69–6.99/GPU-hr
   with 2.75–22 TiB *local* NVMe included. `pod_prep.sh` works nearly unmodified.
5. **Best "platform tooling" alternative** (if we want checkpoint management and
   not just a VM): Baseten Training Jobs (custom containers, B200 $9.98/hr) or
   Modal (B200 $6.25/hr, per-second, up to 3 TiB ephemeral disk + cheap volumes).

## Why managed fine-tuning can't host the pipeline

Our stages vs. what managed FT APIs provide:

| Stage | Needs | Managed FT API |
|---|---|---|
| Regen | vLLM serving the *exact deployed* W4A16 GPTQ quant at SERVING_PARAMS (t 0.7 / top_p 0.95 / top_k 20) | No — their serving stack, their quants |
| DFlash2 training | speculators 0.8.1 + repo patches, layer taps, MTP head, dpace loss pinned, multipack sampler, two venvs (torch 2.13/vllm 0.31, torch 2.14/llmcompressor) | No — canned SFT/DPO/LoRA on standard archs only |
| Eval ladder | our vLLM + τ measurement (n≥4 at t=0.7), spec-decode on/off comparison | No |
| Quant prep | llmcompressor GPTQ + calib cascade-v1 | No |
| Publish | HF Hub | Anywhere |

The drafter is not a fine-tune of a stock model — it's a different architecture
initialized from tapped target layers. Even Together's most flexible managed
surface (fine-tuning API with pre-tokenized Parquet + custom loss masking, or
Post-Training with user-supplied PyTorch loss functions over the model's
log-probs) operates on catalog base models in their runtime. Hard no for DFlash2.

Managed FT *could* still be useful one day for off-policy SFT baselines on
stock Qwen3.5 (see appendix) — but never for the drafter pipeline itself.

## Together.ai — three products, one relevant

### 1. Fine-tuning API (managed) — not suitable
- LoRA + full FT + DPO; billed per token processed; $4 min/job.
- Catalog includes **Qwen3.5 4B, 9B, 27B** (LoRA and full) — our exact family,
  but no 0.8B. BYOM: any HF CausalLM < 100B params, safetensors, no
  `trust_remote_code`, transformers ≤ 5.10, must pair with a catalog template.
- Pre-tokenized Parquet with custom loss masking is supported — the closest any
  managed FT gets to our loss-masking needs. Still standard-arch only.

### 2. Post-Training (custom training loop) — closer, still no
- You write the training loop on your own machine; their "model resource"
  (B200-SXM only, ~$11/GPU-hr implied, billed per GPU-minute) runs the model:
  `sample`, `forward_backward`, `optim_step`, `weights_sync`. Custom loss
  functions over log-probs are supported. Generator replicas for on-policy RL.
- But: catalog-only base models (Qwen3.8-27B, Qwen3.6-35B-A3B, Kimi K3, GLM 5.3,
  MiniMax M3 — **no Qwen3.5 4B/0.8B**), LoRA or full-weight on standard archs,
  their runtime. Can't load a custom drafter, can't serve our W4A16 quant.

### 3. GPU Clusters ("Instant Clusters") — the real option
- Self-serve clusters, **Kubernetes or Slurm-on-K8s**, SSH-accessible Slurm
  login nodes, BYO container images, pinned drivers/CUDA, Grafana observability,
  REST/CLI/Terraform/SkyPilot.
- Persistent shared volumes: min 1 TiB, resizable, survive cluster teardown,
  $0.16/GiB/mo (≈3× RunPod). Local NVMe is ephemeral. `/home` persistent on
  Slurm (NFS), ephemeral on K8s.
- Pricing seen (conflicting across their own pages — verify in console):
  H100 on-demand $5.49 (scale from 8 GPUs), H200 $6.79, **B200 $8.19–9.95
  on-demand** (reserved from ~$6.79–9.09; preemptible preview $4.09). Marketing
  table says B200 clusters scale "256–1000+ GPUs" while on-demand B200s "just
  launched" — whether small (1–8 GPU) on-demand B200 clusters are actually
  creatable is **unverified**. If yes: viable RunPod replacement with a Slurm
  flavor that fits our SSH workflow. If B200 starts at 256 GPUs, only H100 (from
  8 GPUs, $5.49) is lab-sized — and 80 GB H100 is plenty for 4B/0.8B drafter
  training (re-anchor baselines per cross-machine doctrine before comparing τ).

## Alternatives matrix (custom-code capable)

| Platform | Model | B200 $/GPU-hr | Storage story | Workflow fit |
|---|---|---|---|---|
| **RunPod** (current) | VM pod, SSH | $5.98 (Community) / $7.99 (Secure); $4.78 6-mo commit | Network volume ≤ 4 TB, $0.05/GB/mo, DC-pinned, attach at creation; volume disk dies with pod; container disk wiped on stop | Current — pod_prep.sh as-is |
| **Lambda** | VM, SSH, Lambda Stack | $6.99 (1×, 2.75 TiB SSD) → $6.89 (2×, 5.5 TiB) → $6.79 (4×, 11 TiB) → $6.69 (8×, 22 TiB); no egress fees | **Local NVMe included, huge** — the direct disk fix | pod_prep.sh nearly as-is |
| **Modal** | Serverless containers | $6.25, per-second; B300 $7.10 | Volumes $0.09/GiB/mo (v1: 500k inodes; v2 beta: unlimited); **ephemeral disk up to 3 TiB/container** (default 512 GiB) | Real rewrite: Python SDK, no SSH/rsync pod model; `modal shell` helps. Great for burst eval/regen; preemption-aware checkpointing |
| **Baseten Training Jobs** | Job-push containers | $9.98 (also H100 $6.50) | Checkpoints auto-synced to their storage, deployable as endpoints in one command | Custom container + custom code supported ("runs your training code as-is"); workflow = `baseten train push`, not SSH. Multi-node IB |
| **Together GPU Clusters** | Slurm/K8s cluster | $8.19–9.95 (verify min size) | Shared FS $0.16/GiB/mo, min 1 TiB, resizable, survives teardown | SSH login nodes + BYO container = pod-like; storage pricey; B200 min-scale unverified |
| **Crusoe** | VM + K8s/Slurm | ~$8.60–9.65 | Persistent disk $0.08/GiB/mo, object storage $0.06/GiB/mo, zero egress | VM-style, fine but pricier than Lambda for our shape |
| **Vultr** | VM | $3.50 (cheapest tracked; details unverified) | Unverified | Worth a look if B200 availability/disk checks out |

Discarded: Fireworks (managed recipes only / RL platform for their stack),
Nebius Token Factory (managed SFT/LoRA only — raw Nebius VMs exist but Lambda
beats them here), Predibase (LoRA-serving platform, acquired by Cloudflare),
Voltage Park / CoreWeave (cluster-scale commitments, wrong size for us),
hyperscalers (B200 at $8+/hr, 8-GPU minimums, storage + egress nickel-and-dime).

## The disk problem specifically

- **In place (recommended first move):** create a 2–4 TB RunPod network volume in
  the B200 datacenter ($100–200/mo at $0.05/GB), attach at pod creation, move
  HF cache + corpora + checkpoints + venvs there. Note: one network volume per
  pod, DC-pinned (narrows GPU availability to that DC), size increase-only.
  venvs on the volume avoid rebuild churn (container disk is wiped on stop).
- **Lambda:** disk ceases to be a constraint at all — 2.75 TiB local on a 1×B200,
  5.5 TiB on 2×. Fast local NVMe beats any network volume for checkpoint I/O.
- **Modal:** 3 TiB ephemeral per container + volumes for persistence; volume
  bandwidth ~2.5 GB/s (fine for HF cache + corpora; not NVMe-fast).
- **Together:** functional but 3× the storage price; 1 TiB minimum granularity.

## Cost sketch for a 4B-drafter campaign (per B200-hour, on-demand)

RunPod Community 5.98 < Modal 6.25 < Lambda 6.69–6.99 < Together ~8.19–9.95 <
Crusoe ~8.60–9.65 < Baseten 9.98. A 50-GPU-hour campaign differs by ~$200
between cheapest and priciest — **storage ergonomics and workflow fit dominate
the GPU rate at our scale**. (Cross-machine τ doctrine applies regardless:
re-anchor the epoch-0 control and champion on any new box before believing
deltas.)

## Recommendation

1. **Now:** attach a 2–4 TB network volume to the RunPod B200 pod. Cheapest,
   zero workflow change, buys months of headroom.
2. **Parallel-track a Lambda trial:** 1×B200 at $6.99 with 2.75 TiB local NVMe;
   run `pod_prep.sh`, re-anchor epoch-0 + champion τ on it, keep as the
   overflow/burst box. If RunPod DC capacity pinches again, it's the drop-in.
3. **Verify Together cluster sizing** (console: can you actually create an
   on-demand ≤8×B200 cluster?). If yes, it becomes the strongest
   "platform-with-tooling" candidate (Slurm + SSH + BYO container + managed
   observability) — at a storage premium. If no, file Together under
   "interesting at H100 scale only."
4. **Skip** managed FT and Post-Training for DFlash2 — architecturally
   impossible today. Revisit only if they add BYO-architecture training.

## Appendix: managed FT as a *baseline* tool (not DFlash2)

If we ever want cheap off-policy SFT baselines on stock Qwen3.5 (e.g., to
re-measure the rung-1 off-policy artifact against a clean reference):
- Together FT: Qwen3.5-4B LoRA+full, per-token billing, BYOM for 0.8B.
- Crusoe serverless FT: Qwen3.5 4B/9B/2B at $0.40/M training tokens.
- Neither controls data order/packing the way the speculators trainer does, so
  results are suggestive, not comparable to ladder runs — don't quote τ from them.

## Sources (Oct 2026)

- together.ai: /fine-tuning, /gpu-clusters, /pricing; docs.together.ai:
  fine-tuning/{supported-models,pricing,byom}, post-training/{overview,concepts,
  supported-models,pricing}, gpu-clusters-{overview,quickstart}
- lambda.ai/pricing (instance ladder incl. storage), baseten.co/pricing +
  docs.baseten.co/training/overview, modal.com/pricing + docs (volumes,
  resources/disk), crusoe.ai/cloud/pricing, runpod.io/pricing +
  docs.runpod.io/storage/network-volumes
- computeprices.com (B200 cross-provider tracker), northflank.com RunPod pricing
  breakdown (cross-check)
