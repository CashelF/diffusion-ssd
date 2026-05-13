"""Launch a vLLM server and benchmark it.

Handles server lifecycle: launch, health-check, benchmark, cleanup.
The benchmark client (vllm_eval_client.py) sends requests and logs metrics.

Usage:
    python run_vllm_bench.py --llama                      # SD, Llama 70B
    python run_vllm_bench.py --qwen                       # SD, Qwen 32B
    python run_vllm_bench.py --llama --mode ar             # autoregressive baseline
    python run_vllm_bench.py --qwen --mode dflash          # DFlash, Qwen3-8B default
    python run_vllm_bench.py --llama --wandb --name myrun  # log to wandb

Set model paths via env vars (BENCH_LLAMA_70B, etc.) or edit bench_paths.py.
"""
import os
import sys
import json
import time
import signal
import argparse
import subprocess
import requests

sys.path.insert(0, os.path.dirname(__file__))
from bench_paths import MODELS, resolve_snapshot


def get_server_cmd(args):
    if args.target_model is not None:
        target = resolve_snapshot(args.target_model)
    elif args.mode == "dflash":
        target = (
            "meta-llama/Llama-3.1-8B-Instruct"
            if args.llama else
            "Qwen/Qwen3-8B"
        )
    elif args.llama:
        target = resolve_snapshot(MODELS["llama_70b"])
    else:
        target = resolve_snapshot(MODELS["qwen_32b"])

    if args.draft_model is not None:
        draft = resolve_snapshot(args.draft_model)
    elif args.mode == "dflash":
        draft = (
            "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
            if args.llama else
            "z-lab/Qwen3-8B-DFlash-b16"
        )
    elif args.llama:
        draft = resolve_snapshot(MODELS["llama_1b"])
    else:
        draft = resolve_snapshot(MODELS["qwen_0.6b"])

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", target,
        "--tensor-parallel-size", str(args.tp),
        "--gpu-memory-utilization", str(args.mem_frac),
        "--max-num-seqs", "1",
        "--disable-log-requests",
        "--port", str(args.port),
    ]

    if args.mode == "sd":
        # Speculative decoding with standalone draft model.
        spec_config = {
            "model": draft,
            "num_speculative_tokens": args.num_draft_tokens,
            "method": "draft_model",
        }
        cmd += ["--speculative-config", json.dumps(spec_config)]
    elif args.mode == "dflash":
        # DFlash requires vLLM v0.20.1+.
        spec_config = {
            "model": draft,
            "num_speculative_tokens": args.num_draft_tokens,
            "method": "dflash",
        }
        if args.dflash_attention_backend is not None:
            spec_config["attention_backend"] = args.dflash_attention_backend
        cmd += ["--speculative-config", json.dumps(spec_config)]
    # mode == "ar": no speculative flags, just serve the target model.

    return cmd, target


def wait_for_server(port, timeout=900, interval=5):
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(interval)
    return False


def kill_server(proc):
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


def main():
    parser = argparse.ArgumentParser(description="Launch vLLM server and benchmark it")
    parser.add_argument("--llama", action="store_true", default=True)
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--mode", choices=["ar", "sd", "dflash"], default="sd",
                        help="ar = autoregressive, sd = draft-model speculative decoding, dflash = DFlash speculative decoding")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--port", type=int, default=40020)
    parser.add_argument("--mem_frac", type=float, default=0.90)
    parser.add_argument("--num_draft_tokens", type=int, default=None,
                        help="Speculative draft tokens. Defaults to 5 for --mode sd and 15 for --mode dflash.")
    parser.add_argument("--target-model", type=str, default=None,
                        help="Override target model path/HF id. Useful for DFlash models not covered by --llama/--qwen.")
    parser.add_argument("--draft-model", type=str, default=None,
                        help="Override draft model path/HF id. For --mode dflash this should be a z-lab/*-DFlash model.")
    parser.add_argument("--dflash-attention-backend", type=str, default=None,
                        help="Optional DFlash speculative-config attention_backend passed through to vLLM.")
    # Pass-through to eval client
    parser.add_argument("--numseqs", type=int, default=128)
    parser.add_argument("--output_len", type=int, default=512)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    if args.qwen:
        args.llama = False
    if args.num_draft_tokens is None:
        args.num_draft_tokens = 15 if args.mode == "dflash" else 5

    server_cmd, target = get_server_cmd(args)
    print(f"Mode: {args.mode}, Target: {target}")
    print(f"Server cmd: {' '.join(server_cmd)}")

    # Kill stale vllm processes
    subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"],
                   capture_output=True)
    time.sleep(2)

    proc = subprocess.Popen(server_cmd, preexec_fn=os.setsid)
    try:
        print("Waiting for server...")
        if not wait_for_server(args.port):
            print("Server failed to start"); sys.exit(1)
        print("Server ready")

        bench_dir = os.path.dirname(__file__)
        eval_size = "8" if args.mode == "dflash" else ("70" if args.llama else "32")
        eval_cmd = [
            sys.executable, os.path.join(bench_dir, "vllm_eval_client.py"),
            "--size", eval_size,
            "--numseqs", str(args.numseqs),
            "--output_len", str(args.output_len),
            "--temp", str(args.temp),
            "--all", "--b", "1",
            "--port", str(args.port),
            "--model-path", target,
        ]
        if args.llama:
            eval_cmd.append("--llama")
        else:
            eval_cmd.append("--qwen")
        if args.mode == "sd":
            eval_cmd += ["--draft", "1" if args.llama else "0.6"]
        if args.wandb:
            eval_cmd += ["--wandb"]
            if args.group:
                eval_cmd += ["--group", args.group]
            if args.name:
                eval_cmd += ["--name", args.name]

        print(f"Eval cmd: {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd, check=True, cwd=bench_dir)
    finally:
        kill_server(proc)
        print("Server stopped")


if __name__ == "__main__":
    main()
