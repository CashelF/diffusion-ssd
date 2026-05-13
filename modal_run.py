"""
Modal launcher for the V2 DFlash universal-drafter experiment.

Spec: speculative decoding on Qwen3-8B with DFlash as the drafter on BOTH
cache hits AND cache misses. AR target verifies only; DFlash drafts on
both paths (using the cached continuation as a prior on hits and from
all-masks on misses).

Usage on Modal (after `modal token set --token-id ... --token-secret ...`):
    modal run modal_run.py
    modal run modal_run.py --num-seqs 32 --output-len 256
    modal run modal_run.py --no-universal       # baseline: hits=cache, misses=dflash
    modal run modal_run.py --vanilla-ar         # baseline: hits=cache, misses=AR-jit

The function runs on 2x H100 because dflash_ssd needs one GPU each for
target and async draft. Datasets and models are pulled from the
Hugging Face hub at startup. Set HF_TOKEN as a Modal secret if the
checkpoints become gated.
"""

import os
import shlex
import subprocess
from pathlib import Path

import modal

REPO_DIR = "/root/diffusion-ssd"
HF_CACHE = "/cache/huggingface"
DATASET_DIR = "/cache/processed_datasets"

TARGET_MODEL = "Qwen/Qwen3.5-27B"
AR_DRAFT_MODEL = "Qwen/Qwen3-0.6B"
DFLASH_MODEL = "z-lab/Qwen3.5-27B-DFlash"
TARGET_SIZE = "27"  # passed as --size to bench.py

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "libnuma1")
    .pip_install(
        "torch==2.8.0",
        "triton==3.4.0",
        # Bumped from 4.57.1 to git main to pick up Qwen3.5 support
        # (model_type='qwen3_5' isn't in any pinned release yet).
        "transformers @ git+https://github.com/huggingface/transformers.git",
        "xxhash==3.5.0",
        "numpy==2.3.3",
        "safetensors==0.6.2",
        "tqdm==4.67.1",
        "flashinfer-python==0.5.2",
        "sgl-kernel==0.3.17.post1",
        "nvidia-cutlass-dsl==4.2.1",
        "wandb==0.22.0",
        "hf_transfer",
        "tiktoken",
        "datasets",
        "huggingface_hub",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": HF_CACHE,
            "HUGGINGFACE_HUB_CACHE": f"{HF_CACHE}/hub",
            "SSD_HF_CACHE": f"{HF_CACHE}/hub",
            "SSD_DATASET_DIR": DATASET_DIR,
            "SSD_CUDA_ARCH": "9.0",
            "TORCH_CUDA_ARCH_LIST": "9.0",
        }
    )
    # Mount the local repo into the image at /root/diffusion-ssd.
    .add_local_dir(
        local_path=str(Path(__file__).parent),
        remote_path=REPO_DIR,
        ignore=[".venv", "__pycache__", ".git", "report", "*.pyc"],
    )
)

app = modal.App("dflash-ssd-v2", image=image)
hf_volume = modal.Volume.from_name("dflash-hf-cache", create_if_missing=True)


def _download_models():
    from huggingface_hub import snapshot_download

    for repo_id in (TARGET_MODEL, AR_DRAFT_MODEL, DFLASH_MODEL):
        print(f"[modal] ensuring {repo_id} in cache", flush=True)
        snapshot_download(
            repo_id=repo_id,
            cache_dir=f"{HF_CACHE}/hub",
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.txt",
                "*.model",
                "*.py",  # custom modeling code (DFlash uses trust_remote_code)
                "tokenizer*",
            ],
        )


def _prepare_datasets(num_samples: int = 10000):
    """Materialize HumanEval/Alpaca/GSM8K/UltraFeedback prompts.

    bench_helpers.py looks up files by the hardcoded name `*_data_10000.jsonl`,
    so we always materialize that size. The files are small (~few MB each).
    """
    import sys

    sys.path.insert(0, REPO_DIR)
    os.makedirs(DATASET_DIR, exist_ok=True)

    # Skip download if all five datasets already exist at the expected path.
    expected = [
        f"{DATASET_DIR}/{ds}/{ds.split('/')[-1]}_data_{num_samples}.jsonl"
        for ds in ("gsm8k", "humaneval", "alpaca", "c4", "ultrafeedback")
    ]
    if all(os.path.exists(p) for p in expected):
        print(f"[modal] datasets already cached at {DATASET_DIR}", flush=True)
        return

    cmd = [
        "python",
        f"{REPO_DIR}/scripts/get_data_from_hf.py",
        "--num-samples",
        str(num_samples),
    ]
    env = {
        **os.environ,
        "HF_DATASETS_CACHE": "/cache",
        "SSD_DATASET_DIR": DATASET_DIR,
    }
    subprocess.run(cmd, check=True, env=env)


@app.function(
    gpu="H100:4",
    timeout=60 * 60,
    volumes={HF_CACHE: hf_volume},
    secrets=[modal.Secret.from_name("hf-token", required_keys=[])],
)
def bench(
    num_seqs: int = 16,
    output_len: int = 256,
    k: int = 15,
    dataset: str = "gsm8k",
    universal: bool = True,
    skip_tree_cache: bool = False,
    extra_args: str = "",
):
    """Launch the bench script inside the Modal container.

    `dataset` is one of: gsm8k (default), humaneval, alpaca, c4, ultrafeedback, all.
    `universal=True` (default) → V2: DFlash on both hits and misses.
    `universal=False` → V1 baseline: cached tokens on hits, DFlash on misses.
    """
    _download_models()
    # bench_helpers.py hardcodes the dataset filename to *_data_10000.jsonl,
    # so we always materialize that exact size (files are small and cached
    # in the persistent Modal volume after the first run).
    _prepare_datasets(num_samples=10000)

    # In bench.py, GSM8K is the implicit default when no other dataset flag
    # is set; the other datasets have explicit flags. `--all` runs all four.
    dataset_flag = {
        "gsm8k": None,
        "humaneval": "--humaneval",
        "alpaca": "--alpaca",
        "c4": "--c4",
        "ultrafeedback": "--ultrafeedback",
        "all": "--all",
    }[dataset]

    # Resolve DFlash checkpoint snapshot path under the HF hub cache.
    snapshots = Path(
        f"{HF_CACHE}/hub/models--{DFLASH_MODEL.replace('/', '--')}/snapshots"
    )
    dflash_path = next(snapshots.iterdir())

    cmd = [
        "python",
        "-O",
        f"{REPO_DIR}/bench/bench.py",
        "--qwen",
        "--size",
        TARGET_SIZE,
        "--spec",
        "--async",
        "--draft-backend",
        "dflash_ssd",
        "--draft",
        "0.6",
        "--dflash-draft",
        str(dflash_path),
        "--k",
        str(k),
        "--gpus",
        "4",
        "--b",
        "1",
        "--temp",
        "0",
        "--dtemp",
        "0",
        "--numseqs",
        str(num_seqs),
        "--output_len",
        str(output_len),
    ]
    if dataset_flag is not None:
        cmd.append(dataset_flag)
    if universal:
        cmd.append("--dflash-universal-drafter")
    if skip_tree_cache:
        cmd.append("--dflash-ssd-skip-tree-cache")
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    print("[modal] launching:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "PYTHONPATH": f"{REPO_DIR}:{os.environ.get('PYTHONPATH', '')}",
    }
    subprocess.run(cmd, check=True, cwd=REPO_DIR, env=env)


@app.local_entrypoint()
def main(
    num_seqs: int = 16,
    output_len: int = 256,
    k: int = 15,
    dataset: str = "gsm8k",
    no_universal: bool = False,
    skip_tree_cache: bool = False,
    extra_args: str = "",
):
    bench.remote(
        num_seqs=num_seqs,
        output_len=output_len,
        k=k,
        dataset=dataset,
        universal=not no_universal,
        skip_tree_cache=skip_tree_cache,
        extra_args=extra_args,
    )
