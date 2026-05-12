# Presentation Plan — Diffusion-SSD Basic Results

Goal: by tomorrow, have a small but credible benchmark that compares
**AR-draft speculative decoding** vs **diffusion-draft speculative decoding**
on the same target model and same GPU class as the SSD paper, so we can
say with numbers whether the diffusion draft makes generation faster.

---

## 1. Reality check on "same model + same GPU as the paper"

The paper benchmarks `Llama-3.3-70B` (target) + `Llama-3.2-1B` (AR draft) on
**H100s** (4–5 of them).

Two problems with replicating it exactly tomorrow:

1. **No Llama-based diffusion draft exists.** The only diffusion draft
   wired into this repo today is `dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`.
   So we cannot do `Llama-70B target + diffusion draft` at all — the
   apples-to-apples diffusion comparison has to use Qwen.
2. **Cost / time.** 4×H100 on Modal is ~$10–15/hr; downloading a 70B
   checkpoint is ~140 GB and slow even on Modal's network. Doable for one
   short run, not for iteration.

**Recommended compromise** (this is what the rest of this doc assumes):

- Match the paper's **GPU class** (H100) but use **1 GPU**.
- Match the paper's **methodology** (sync spec decode, k=4–7, temp=0,
  4 datasets via `--all`, 128 prompts/dataset) but at a smaller model
  scale where a diffusion draft actually exists: **Qwen3-1.7B target +
  Qwen3-0.6B draft**, AR vs BD3LM.
- Optional stretch: one extra run reproducing the paper's Llama 70B + 1B
  AR baseline on 4×H100, just to anchor that our setup matches theirs.

That gives a clean 3-bar chart for the slide: AR-only baseline / sync-SD with
AR draft / sync-SD with diffusion draft, all at the same target model +
same hardware.

> Important constraint: `--draft-backend block` (the diffusion path) only
> supports **sync** speculation today, not async (`--async`). So our
> diffusion run is sync SD with a diffusion draft, not full SSD. This is
> fine for "does diffusion drafting help?" but be honest about it in the
> slide.

---

## 2. Modal setup (one-time, do this first)

Estimated time: 30–45 min including the Qwen downloads.

### 2a. Auth
```bash
pip install modal
modal token set --token-id <ID> --token-secret <SECRET>
```

### 2b. App file
Create `modal_bench.py` in the repo root (`code/diffusion-ssd/`). Skeleton:

```python
import modal

app = modal.App("diffusion-ssd-bench")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:$PATH"})
    .add_local_dir(".", remote_path="/root/diffusion-ssd", copy=True)
    .workdir("/root/diffusion-ssd")
    .run_commands("uv sync --extra scripts && uv pip install huggingface_hub[cli]")
    .env({
        "SSD_HF_CACHE": "/root/hf_cache/hub",
        "SSD_DATASET_DIR": "/root/datasets",
        "HF_DATASETS_CACHE": "/root",
        "SSD_CUDA_ARCH": "9.0",
    })
)

hf_vol = modal.Volume.from_name("ssd-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("ssd-datasets", create_if_missing=True)

VOLUMES = {"/root/hf_cache": hf_vol, "/root": data_vol}

@app.function(image=image, gpu="H100", timeout=60*60, volumes=VOLUMES,
              secrets=[modal.Secret.from_name("huggingface")])  # only needed for Llama
def download_qwen():
    import subprocess
    subprocess.run(["uv", "run", "python", "scripts/download_from_hf.py", "qwen"], check=True)
    # BD3LM diffusion draft
    subprocess.run([
        "uv", "run", "huggingface-cli", "download",
        "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1",
        "--cache-dir", "/root/hf_cache/hub",
    ], check=True)
    subprocess.run(["uv", "run", "python", "scripts/get_data_from_hf.py",
                    "--num-samples", "256"], check=True)

@app.function(image=image, gpu="H100", timeout=60*60, volumes=VOLUMES)
def run_bench(args: list[str]):
    import subprocess
    subprocess.run(["uv", "run", "python", "-O", "bench/bench.py", *args],
                   check=True, cwd="/root/diffusion-ssd")
```

### 2c. HF token (only if doing the Llama stretch run)
Create a Modal secret called `huggingface` containing `HF_TOKEN=<your token>`.
You also need to have accepted the Llama license on HF.

### 2d. First-time download
```bash
modal run modal_bench.py::download_qwen
```
This caches Qwen3-1.7B, Qwen3-0.6B, the BD3LM draft, and the datasets into
the persistent volume. Subsequent runs are instant.

---

## 3. The three benchmark runs

All three target the same model (Qwen3-1.7B), same hardware (1×H100),
same data (`--all`, 128 prompts/dataset, 512 output tokens, temp=0).

### Run A — AR baseline (no speculation)
```bash
modal run modal_bench.py::run_bench --args '["--qwen","--size","1.7","--gpus","1","--b","1","--temp","0","--numseqs","128","--output_len","512","--all","--name","A_ar_baseline"]'
```

### Run B — Sync SD, AR draft (Qwen3-0.6B)
```bash
modal run modal_bench.py::run_bench --args '["--qwen","--size","1.7","--spec","--draft","0.6","--k","6","--gpus","1","--b","1","--temp","0","--numseqs","128","--output_len","512","--all","--name","B_sd_ar_draft"]'
```

### Run C — Sync SD, diffusion draft (BD3LM Qwen3-0.6B)
```bash
modal run modal_bench.py::run_bench --args '["--qwen","--size","1.7","--spec","--draft-backend","block","--draft","/root/hf_cache/hub/models--dllm-hub--Qwen3-0.6B-diffusion-bd3lm-v0.1","--k","15","--block-refine-steps","2","--block-sampler","remask","--block-special-tokens","interior","--gpus","1","--b","1","--temp","0","--numseqs","128","--output_len","512","--all","--name","C_sd_diffusion_draft"]'
```

> If C errors on the path, resolve the snapshot dir under
> `models--dllm-hub--Qwen3-0.6B-diffusion-bd3lm-v0.1/snapshots/<hash>/` and
> pass that. `bench/compare_speculative_drafts.py:_default_block_draft`
> shows the resolution logic.

Each `--all` run is 4×128 = 512 prompts. Expect ~10–20 min per run on 1×H100;
budget ~1 hour total for all three.

### Optional stretch — paper anchor
One Llama 70B AR baseline + sync-SD AR-draft pair on 4×H100, just to
confirm we see the same ballpark speedup the paper reports (~2×). Only do
this if A/B/C are done and you have budget left:
```bash
modal run modal_bench.py::run_bench --args '["--llama","--size","70","--gpus","4","--b","1","--temp","0","--numseqs","32","--output_len","512","--all","--name","Stretch_llama70_ar"]'
modal run modal_bench.py::run_bench --args '["--llama","--size","70","--gpus","4","--spec","--k","6","--b","1","--temp","0","--numseqs","32","--output_len","512","--all","--name","Stretch_llama70_sd"]'
```
(Lower `--numseqs` to 32 to keep it cheap. Use `gpu="H100:4"` in the Modal
function for this one.)

---

## 4. What to extract for the slide

`bench.py` writes a JSON result file per run. From each, pull:

- **tokens/sec** (decode throughput, the headline number)
- **mean acceptance length / mean accepted tokens per draft step** (for B and C)
- **per-dataset breakdown** (humaneval / alpaca / gsm8k / ultrafeedback) — the
  story is usually different by dataset, and that variation is itself the
  finding.

Slide layout suggestion:

1. One bar chart: tokens/sec for A / B / C, with the speedup-over-A ratio
   labeled on each bar.
2. One small table: per-dataset acceptance length for B vs C. If the
   diffusion draft has higher acceptance on code-y datasets (humaneval,
   gsm8k) than on chat (alpaca, ultrafeedback), say so — that's the
   interesting result.
3. One slide of caveats: sync-only (no async/SSD yet for the diffusion
   path), 1.7B target not 70B, single GPU not 4. State why (Section 1).

---

## 5. Order of operations for tonight

1. `modal token set ...` and verify with `modal app list` (~2 min)
2. Drop in `modal_bench.py`, run `download_qwen` (~20–30 min, mostly idle)
3. Kick off Run A. While it runs, write the slide skeleton.
4. Run B, then Run C.
5. Pull JSONs locally with `modal volume get ssd-datasets ...` or just
   print them at the end of `run_bench`. Build the chart.
6. (Optional) stretch Llama runs if it's still early.

If anything blocks for >20 min, drop the stretch run and ship the
Qwen-only result. That's still a real apples-to-apples
"diffusion-vs-AR draft, same target, same GPU class as the paper"
comparison, which is the actual claim we want to make.
