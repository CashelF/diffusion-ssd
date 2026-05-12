"""Compare native greedy AR with native synchronous DFlash speculation.

Each mode runs in a child process so CUDA state is released between runs. The
comparison checks exact greedy parity and reports DFlash acceptance metrics.
"""

import argparse
import json
import os
import subprocess
import sys
from time import perf_counter
from typing import Any


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bench_paths import HF_CACHE_DIR, resolve_snapshot


RESULT_PREFIX = "__RESULT_JSON__="


def _default_target() -> str:
    return resolve_snapshot(
        os.environ.get(
            "BENCH_QWEN_TARGET_8B",
            f"{HF_CACHE_DIR}/models--Qwen--Qwen3-8B",
        )
    )


def _default_dflash_draft() -> str:
    return resolve_snapshot(
        os.environ.get(
            "BENCH_QWEN_DFLASH_8B_B16",
            f"{HF_CACHE_DIR}/models--z-lab--Qwen3-8B-DFlash-b16",
        )
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compare", "ar", "dflash"], default="compare")
    parser.add_argument("--target", type=str, default=_default_target())
    parser.add_argument("--dflash-draft", type=str, default=_default_dflash_draft())
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--user-prompt", type=str, default="Introduce yourself briefly.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--speculate-k", type=int, default=15)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--dflash-gpu-memory-reserve-gb", type=float, default=3.0)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--verbose-engine", action="store_true")
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def _accepted_fraction(accepted_suffix_lens: list[int], speculate_k: int) -> float | None:
    if not accepted_suffix_lens:
        return None
    avg_tokens_per_step = sum(accepted_suffix_lens) / len(accepted_suffix_lens)
    return (avg_tokens_per_step - 1.0) / speculate_k


def _run_single_mode(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    import ssd.paths  # noqa: F401
    from ssd import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.target, use_fast=True)
    prompt_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.user_prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    print(f"=== {mode.upper()} RUN ===", flush=True)
    print(f"TARGET {args.target}", flush=True)
    if mode == "dflash":
        print(f"DFLASH_DRAFT {args.dflash_draft}", flush=True)
    print(f"PROMPT_LEN {len(prompt_ids)}", flush=True)

    llm_kwargs = dict(
        num_gpus=1,
        enforce_eager=args.eager,
        max_num_seqs=1,
        max_model_len=args.max_model_len,
        kvcache_block_size=args.kvcache_block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        verbose=args.verbose_engine,
    )
    if mode == "dflash":
        llm_kwargs.update(
            draft=args.dflash_draft,
            speculate=True,
            draft_backend="dflash",
            speculate_k=args.speculate_k,
            dflash_gpu_memory_reserve_gb=args.dflash_gpu_memory_reserve_gb,
        )

    llm = LLM(args.target, **llm_kwargs)
    t0 = perf_counter()
    try:
        outputs, metrics = llm.generate(
            [prompt_ids],
            [SamplingParams(
                temperature=0.0,
                draft_temperature=0.0,
                ignore_eos=True,
                max_new_tokens=args.max_new_tokens,
            )],
            use_tqdm=False,
        )
    finally:
        llm.exit(hard=False)
    total_time = perf_counter() - t0

    output = outputs[0]
    accepted_suffix = metrics.get("accepted_suffix_lens_with_recovery", [])
    decode_tokens = len(output["token_ids"])
    result = {
        "mode": mode,
        "target": args.target,
        "dflash_draft": args.dflash_draft if mode == "dflash" else None,
        "prompt_len": len(prompt_ids),
        "decode_tokens": decode_tokens,
        "total_time_s": total_time,
        "end_to_end_tok_s": decode_tokens / total_time if total_time > 0 else None,
        "token_ids": output["token_ids"],
        "text": output["text"],
        "accepted_suffix_lens_with_recovery": accepted_suffix,
        "avg_tokens_per_step_incl_recovery": (
            sum(accepted_suffix) / len(accepted_suffix) if accepted_suffix else None
        ),
        "avg_accepted_speculative_fraction": _accepted_fraction(
            accepted_suffix,
            args.speculate_k,
        ),
    }
    print(RESULT_PREFIX + json.dumps(result), flush=True)
    return result


def _child_args(args: argparse.Namespace, mode: str) -> list[str]:
    cmd = [
        sys.executable,
        "-O",
        os.path.abspath(__file__),
        "--mode", mode,
        "--target", args.target,
        "--dflash-draft", args.dflash_draft,
        "--system-prompt", args.system_prompt,
        "--user-prompt", args.user_prompt,
        "--max-new-tokens", str(args.max_new_tokens),
        "--speculate-k", str(args.speculate_k),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--dflash-gpu-memory-reserve-gb", str(args.dflash_gpu_memory_reserve_gb),
        "--max-model-len", str(args.max_model_len),
        "--kvcache-block-size", str(args.kvcache_block_size),
    ]
    if args.eager:
        cmd.append("--eager")
    if args.verbose_engine:
        cmd.append("--verbose-engine")
    return cmd


def _run_child(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    proc = subprocess.Popen(
        _child_args(args, mode),
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"{mode} child run failed with exit code {returncode}")

    result_line = next(
        (line for line in reversed(lines) if line.startswith(RESULT_PREFIX)),
        None,
    )
    if result_line is None:
        raise RuntimeError(f"{mode} child run did not emit a result summary")
    return json.loads(result_line[len(RESULT_PREFIX):])


def _compare(args: argparse.Namespace) -> dict[str, Any]:
    ar = _run_child(args, "ar")
    dflash = _run_child(args, "dflash")
    summary = {
        "target": args.target,
        "dflash_draft": args.dflash_draft,
        "same_final_token_ids": ar["token_ids"] == dflash["token_ids"],
        "same_final_text": ar["text"] == dflash["text"],
        "speedup_vs_ar": (
            dflash["end_to_end_tok_s"] / ar["end_to_end_tok_s"]
            if ar["end_to_end_tok_s"] and dflash["end_to_end_tok_s"]
            else None
        ),
        "ar": ar,
        "dflash": dflash,
    }
    print("=== COMPARISON SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.mode == "compare":
        _compare(args)
        return
    _run_single_mode(args, args.mode)


if __name__ == "__main__":
    main()
