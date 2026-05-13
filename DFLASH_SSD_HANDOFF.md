# DFlash + SSD Handoff

This repo is on branch `main` at base commit `cda14f9` with a staged port for
DFlash-backed speculative decoding and DFlash-seeded SSD.

## What Was Implemented

- Added `draft_backend="dflash"` for synchronous DFlash speculative decoding.
- Added `draft_backend="dflash_ssd"` for async SSD where DFlash seeds cache
  misses with the first block and the normal SSD drafter populates/speculates
  the rest of the tree.
- DFlash sync path uses fresh verifier activations from the target model,
  calls a matched DFlash checkpoint, and reuses the existing exact verifier for
  accept/reject.
- Initial supported target is Qwen3, especially:
  - target: `Qwen/Qwen3-8B`
  - DFlash draft: `z-lab/Qwen3-8B-DFlash-b16`
  - SSD/AR draft: `Qwen/Qwen3-0.6B`
- v1 constraints are intentional:
  - DFlash sync: single GPU, batch size 1, greedy only.
  - DFlash+SSD: exactly 2 GPUs, batch size 1, greedy only, async SSD only.

## Key Files

- `ssd/config.py`: config surface, validation, DFlash checkpoint config load,
  `block_size - 1 == speculate_k` enforcement for DFlash.
- `ssd/engine/model_runner.py`: Qwen target activation capture, using
  post-layer residual states matching HF `hidden_states[layer_id + 1]`
  semantics; added `before_allocate_kv_cache()` hook.
- `ssd/engine/draft_backends.py`: DFlash draft backend and DFlash seed helper.
- `ssd/engine/draft_runner.py`: DFlash+SSD miss seeding, async SSD integration,
  DFlash model loaded before draft KV allocation.
- `ssd/engine/speculator_async.py`: DFlash+SSD request/activation handoff.
- `ssd/engine/scheduler.py`: post-verify DFlash activation backlog updates.
- `ssd/engine/llm_engine.py`, `ssd/engine/step.py`, `ssd/engine/sequence.py`:
  plumbing for DFlash activations/results.
- `ssd/engine/helpers/cudagraph_helpers.py`: graph capture uses the correct
  model runner HF config to avoid target/draft hidden-size mismatch.
- `bench/bench.py`, `bench/bench_helpers.py`: CLI flags and path resolution.
- `bench/compare_dflash_native.py`: native DFlash diagnostic.
- `bench/compare_dflash_parity.py`: compares native DFlash backend tensors
  against upstream/HF DFlash behavior.
- `bench/compare_dflash_upstream.py`: compares native runs against upstream
  vLLM DFlash where available.
- `bench/run_vllm_bench.py`, `bench/vllm_eval_client.py`: DFlash-related
  vLLM comparison support.
- `README.md`, `bench/README.md`: usage notes.

## Important CLI Flags

- `--draft-backend dflash`
- `--draft-backend dflash_ssd`
- `--draft <path-or-hf-id>` for sync DFlash, where this is the DFlash checkpoint.
- `--dflash-draft <path-or-hf-id>` for DFlash+SSD, where this is the DFlash
  checkpoint and `--draft` remains the SSD/AR draft model.
- `--dflash-block-size`, defaulting from the DFlash checkpoint.
- `--dflash-mask-token-id`, defaulting from the DFlash checkpoint.
- `--dflash-gpu-memory-reserve-gb`
- `--dflash-ssd-skip-tree-cache` for diagnostics.

## Local Model Paths Used

Set:

```bash
export SSD_HF_CACHE=/data/cashel-data/huggingface
export SSD_DATASET_DIR=/data/cashel-data/ml-data/processed_datasets
```

Reference snapshots used locally:

```text
/data/cashel-data/huggingface/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
/data/cashel-data/huggingface/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
/data/cashel-data/huggingface/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4
```

On A6000, also set:

```bash
export SSD_CUDA_ARCH=8.6
```

On H100/H200, use:

```bash
export SSD_CUDA_ARCH=9.0
```

## Example Commands

Sync DFlash smoke:

```bash
CUDA_VISIBLE_DEVICES=0 python -O bench/bench.py \
  --qwen --size 8 \
  --spec --draft-backend dflash \
  --draft /data/cashel-data/huggingface/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4 \
  --k 15 --gpus 1 --b 1 \
  --temp 0 --dtemp 0 \
  --numseqs 1 --output_len 32 --max-steps 1
```

DFlash+SSD smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -O bench/bench.py \
  --qwen --size 8 \
  --spec --async --draft-backend dflash_ssd \
  --draft 0.6 \
  --dflash-draft /data/cashel-data/huggingface/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4 \
  --k 15 --gpus 2 --b 1 \
  --temp 0 --dtemp 0 \
  --numseqs 1 --output_len 32 --max-steps 1
```

Representative DFlash+SSD run shape:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -O bench/bench.py \
  --qwen --size 8 \
  --spec --async --draft-backend dflash_ssd \
  --draft 0.6 \
  --dflash-draft /data/cashel-data/huggingface/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4 \
  --k 15 --gpus 2 --b 1 \
  --temp 0 --dtemp 0 \
  --numseqs 4 --output_len 64
```

Diagnostic seed-only path:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -O bench/bench.py \
  --qwen --size 8 \
  --spec --async --draft-backend dflash_ssd \
  --draft 0.6 \
  --dflash-draft /data/cashel-data/huggingface/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/9b41424b7109f9c5413454f481b09a82b85333f4 \
  --dflash-ssd-skip-tree-cache \
  --k 15 --gpus 2 --b 1 \
  --temp 0 --dtemp 0 \
  --numseqs 1 --output_len 32 --max-steps 1
```

## Validation Already Done

- `python -m py_compile` / compile checks passed for changed Python files.
- `git diff --check` passed before the handoff file was added.
- `bench/bench.py --help` exposes:
  - `--draft-backend {ar,block,dflash,dflash_ssd}`
  - `--dflash-block-size`
  - `--dflash-mask-token-id`
  - `--dflash-gpu-memory-reserve-gb`
  - `--dflash-draft`
  - `--dflash-ssd-skip-tree-cache`
- Sync DFlash path produced the expected tensor shapes in local diagnostics:
  - `draft_tokens.shape == [1, 15]`
  - `logits_q.shape == [1, 15, vocab]`
- DFlash+SSD normal run completed locally for 32 tokens with:
  - throughput about `11.68 tok/s`
  - average accepted tokens about `3.30`
  - cache hit rate about `0.60`
  - average DFlash seed time about `34.48ms`
  - SSD tree decode about `79ms`

These numbers were not considered final paper-comparison numbers because the
available GPUs were busy and the run was small.

## Blade3 Status

Repo and env were copied to:

```text
cash@192.168.100.19:~/diffusion-ssd
```

The copied env is project-local:

```text
~/diffusion-ssd/.conda
```

Verified there:

```text
Python 3.11.15
torch 2.8.0+cu128
transformers 4.57.1
flashinfer 0.5.2
CUDA devices visible: 8
```

Blade3 was not usable for model runs at handoff time because all A6000s were
occupied by `/home/thomas/anaconda3/envs/rfdetr/bin/python`, with roughly
39-43GB allocated per GPU and most GPUs at 100% utilization.

## Known Risks / Next Things To Pinpoint

- Performance is not yet close enough to the DFlash paper to draw conclusions.
  Current small runs are more useful for correctness and profiling than for
  throughput claims.
- Need a representative run on less busy GPUs, ideally using the same GPU class
  as the paper if possible. A6000 vs H100 can be a large part of the gap, but
  the observed numbers are low enough that profiling is still warranted.
- Need isolate whether async SSD tree decode overhead dominates after DFlash
  seeding. Use `--dflash-ssd-skip-tree-cache` to separate seed cost from tree
  population cost.
- Need compare:
  - AR baseline
  - sync DFlash
  - normal async SSD
  - DFlash+SSD
  under the same prompt/output settings.
- Need confirm verifier and drafter GPU placement on the new machine. For
  `--gpus 2`, use `CUDA_VISIBLE_DEVICES=<verifier_gpu>,<drafter_gpu>` and
  inspect logs / `nvidia-smi` to confirm memory placement.
- For DFlash+SSD, DFlash produces the first block on cache misses from verifier
  activations; normal SSD then continues speculative tree work for subsequent
  branches/blocks.
- Normal SSD block size can be reduced independently for diagnostics, but the
  DFlash checkpoint used here has `block_size=16`, so DFlash v1 requires
  `--k 15`.

## Files Intentionally Not Staged

- `.env`
- `.codex`

These are local/private operational files and should not be needed by the next
Codex instance.
