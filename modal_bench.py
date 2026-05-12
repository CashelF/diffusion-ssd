import modal

app = modal.App("diffusion-ssd-bench")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "curl", "libnuma1", "libnuma-dev")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:/usr/local/cuda/bin:${PATH}"})
    .add_local_dir(
        ".",
        remote_path="/root/diffusion-ssd",
        copy=True,
        ignore=[".git", ".git/**", ".venv", ".venv/**", "__pycache__", "**/__pycache__/**", "*.pyc", "results", "results/**"],
    )
    .workdir("/root/diffusion-ssd")
    .run_commands(
        "cd /root/diffusion-ssd && uv sync --extra scripts",
        "cd /root/diffusion-ssd && uv pip install 'huggingface_hub[cli]'",
    )
    .env({
        "SSD_HF_CACHE": "/root/hf_cache/hub",
        "SSD_DATASET_DIR": "/root/datasets/processed_datasets",
        "HF_DATASETS_CACHE": "/root/datasets",
        "SSD_CUDA_ARCH": "9.0",
    })
)

hf_vol = modal.Volume.from_name("ssd-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("ssd-datasets", create_if_missing=True)
results_vol = modal.Volume.from_name("ssd-results", create_if_missing=True)

VOLUMES = {
    "/root/hf_cache": hf_vol,
    "/root/datasets": data_vol,
    "/root/results": results_vol,
}


@app.function(image=image, gpu="H100", timeout=60 * 60, volumes=VOLUMES)
def download_qwen():
    import subprocess
    subprocess.run(
        ["uv", "run", "python", "scripts/download_from_hf.py", "qwen"],
        check=True, cwd="/root/diffusion-ssd",
    )
    subprocess.run(
        [
            "uv", "run", "huggingface-cli", "download",
            "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1",
            "--cache-dir", "/root/hf_cache/hub",
        ],
        check=True, cwd="/root/diffusion-ssd",
    )
    subprocess.run(
        ["uv", "run", "python", "scripts/get_data_from_hf.py", "--num-samples", "10000"],
        check=True, cwd="/root/diffusion-ssd",
    )
    hf_vol.commit()
    data_vol.commit()


@app.function(image=image, gpu="H100", timeout=2 * 60 * 60, volumes=VOLUMES)
def run_bench(args: list[str], log_name: str):
    import subprocess, shutil, glob, os, sys
    os.makedirs("/root/results", exist_ok=True)
    log_path = f"/root/results/{log_name}.log"
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            ["uv", "run", "python", "-O", "bench/bench.py", *args],
            cwd="/root/diffusion-ssd",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()
    for f in glob.glob("/root/diffusion-ssd/bench/results/*"):
        shutil.copy(f, "/root/results/")
    for f in glob.glob("/root/diffusion-ssd/results/*"):
        shutil.copy(f, "/root/results/")
    results_vol.commit()
    if proc.returncode != 0:
        sys.exit(proc.returncode)


@app.function(image=image, volumes={"/root/results": results_vol}, timeout=300)
def show_results():
    import os, re
    for fn in sorted(os.listdir("/root/results")):
        if not fn.endswith(".log"):
            continue
        path = f"/root/results/{fn}"
        with open(path) as f:
            text = f.read()
        print(f"\n=== {fn} ===")
        for pat in [
            r"Final Prefill Throughput.*",
            r"Final Decode Throughput.*",
            r"Total Throughput.*",
            r"Mode:.*Total:.*Time:.*Throughput:.*",
            r".*acceptance.*",
            r".*accepted.*suffix.*",
        ]:
            for m in re.findall(pat, text, re.IGNORECASE):
                print(m)


@app.local_entrypoint()
def main(stage: str = "download"):
    if stage == "download":
        download_qwen.remote()
    elif stage == "A":
        run_bench.remote([
            "--qwen", "--size", "1.7", "--gpus", "1",
            "--b", "1", "--temp", "0",
            "--numseqs", "128", "--output_len", "512", "--all",
            "--name", "A_ar_baseline",
        ], "A_ar_baseline")
    elif stage == "B":
        run_bench.remote([
            "--qwen", "--size", "1.7", "--spec",
            "--draft", "0.6", "--k", "6", "--gpus", "1",
            "--b", "1", "--temp", "0",
            "--numseqs", "128", "--output_len", "512", "--all",
            "--name", "B_sd_ar_draft",
        ], "B_sd_ar_draft")
    elif stage == "C":
        run_bench.remote([
            "--qwen", "--size", "1.7", "--spec",
            "--draft-backend", "block",
            "--draft", "/root/hf_cache/hub/models--dllm-hub--Qwen3-0.6B-diffusion-bd3lm-v0.1",
            "--k", "15", "--block-refine-steps", "4",
            "--block-sampler", "first_hitting",
            "--block-attention", "staircase",
            "--block-prefix-cache",
            "--block-draft-block-size", "32",
            "--block-special-tokens", "interior",
            "--gpus", "1",
            "--b", "1", "--temp", "0",
            "--numseqs", "32", "--output_len", "512", "--all",
            "--name", "C_sd_diffusion_draft",
        ], "C_sd_diffusion_draft")
    elif stage == "A8":
        run_bench.remote([
            "--qwen", "--size", "8", "--gpus", "1",
            "--b", "1", "--temp", "0",
            "--numseqs", "64", "--output_len", "512", "--all",
            "--name", "A8_ar_baseline_qwen8b",
        ], "A8_ar_baseline_qwen8b")
    elif stage == "B8":
        run_bench.remote([
            "--qwen", "--size", "8", "--spec",
            "--draft", "0.6", "--k", "6", "--gpus", "1",
            "--b", "1", "--temp", "0",
            "--numseqs", "64", "--output_len", "512", "--all",
            "--name", "B8_sd_ar_draft_qwen8b",
        ], "B8_sd_ar_draft_qwen8b")
    elif stage == "C8":
        run_bench.remote([
            "--qwen", "--size", "8", "--spec",
            "--draft-backend", "block",
            "--draft", "/root/hf_cache/hub/models--dllm-hub--Qwen3-0.6B-diffusion-bd3lm-v0.1",
            "--k", "15", "--block-refine-steps", "4",
            "--block-sampler", "first_hitting",
            "--block-attention", "staircase",
            "--block-prefix-cache",
            "--block-draft-block-size", "32",
            "--block-special-tokens", "interior",
            "--gpus", "1",
            "--b", "1", "--temp", "0",
            "--numseqs", "16", "--output_len", "512", "--all",
            "--name", "C8_sd_diffusion_draft_qwen8b",
        ], "C8_sd_diffusion_draft_qwen8b")
    elif stage == "results":
        show_results.remote()
    else:
        raise SystemExit(f"unknown stage: {stage} (use download | A | B | C | A8 | B8 | C8 | results)")
