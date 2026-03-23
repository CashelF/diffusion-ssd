"""Compare AR and block-diffusion speculative drafts on the same prompt.

Runs each backend in a separate child process so GPU state is released cleanly
between runs on single-GPU machines.
"""
import argparse
import json
import os
import subprocess
import sys
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
            "BENCH_QWEN_TARGET_17B",
            f"{HF_CACHE_DIR}/models--Qwen--Qwen3-1.7B",
        )
    )


def _default_ar_draft() -> str:
    return resolve_snapshot(
        os.environ.get(
            "BENCH_QWEN_AR_DRAFT_06B",
            f"{HF_CACHE_DIR}/models--Qwen--Qwen3-0.6B",
        )
    )


def _resolve_existing_path(candidates: list[str]) -> str:
    for candidate in candidates:
        if not candidate:
            continue
        resolved = resolve_snapshot(candidate)
        if os.path.exists(os.path.join(resolved, "config.json")):
            return resolved
    fallback = next((candidate for candidate in candidates if candidate), None)
    if fallback is None:
        raise ValueError("No candidate model paths were provided")
    return resolve_snapshot(fallback)


def _default_block_draft() -> str:
    hf_root = os.path.dirname(HF_CACHE_DIR.rstrip("/"))
    return _resolve_existing_path(
        [
            os.environ.get("BENCH_QWEN_BLOCK_DRAFT_BD3LM"),
            f"{hf_root}/models--dllm-hub--Qwen3-0.6B-diffusion-bd3lm-v0.1",
            f"{HF_CACHE_DIR}/models--dllm-hub--Qwen3-0.6B-diffusion-bd3lm-v0.1",
        ]
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["compare", "ar", "block"],
        default="compare",
        help="Run both backends or a single backend.",
    )
    parser.add_argument("--target", type=str, default=_default_target())
    parser.add_argument("--ar-draft", type=str, default=_default_ar_draft())
    parser.add_argument("--block-draft", type=str, default=_default_block_draft())
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful AI assistant.",
    )
    parser.add_argument(
        "--user-prompt",
        type=str,
        default="Implement a DFS traversal in Python with clear inline comments.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--speculate-k", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.05)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--kvcache-block-size", type=int, default=64)
    parser.add_argument("--block-refine-steps", type=int, default=4)
    parser.add_argument(
        "--block-sampler",
        choices=["mask_predict", "remask", "first_hitting"],
        default="mask_predict",
    )
    parser.add_argument(
        "--block-attention",
        choices=["full", "staircase"],
        default="full",
    )
    parser.add_argument(
        "--block-prefix-cache",
        action="store_true",
    )
    parser.add_argument(
        "--block-draft-block-size",
        type=int,
        default=None,
    )
    parser.add_argument("--block-mask-token-id", type=int, default=None)
    parser.add_argument(
        "--block-special-tokens",
        choices=["none", "interior", "all"],
        default="none",
    )
    parser.add_argument(
        "--block-forbid-token-ids",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--block-no-reuse-step-buffers",
        action="store_true",
    )
    parser.add_argument(
        "--block-warm-start-mode",
        choices=["none", "ar"],
        default="none",
    )
    parser.add_argument("--block-warm-start-tokens", type=int, default=0)
    parser.add_argument("--block-warm-start-draft", type=str, default=None)
    parser.add_argument(
        "--block-warm-start-keep-loaded",
        action="store_true",
    )
    parser.add_argument(
        "--verbose-engine",
        action="store_true",
        help="Enable engine-level speculation logs.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write the combined comparison summary as JSON.",
    )
    return parser


def _accepted_fraction(accepted_suffix_lens: list[int], speculate_k: int) -> float | None:
    if not accepted_suffix_lens:
        return None
    avg_tokens_per_step = sum(accepted_suffix_lens) / len(accepted_suffix_lens)
    return (avg_tokens_per_step - 1.0) / speculate_k


def _run_single_backend(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    from transformers import AutoTokenizer

    import ssd.paths  # noqa: F401
    from ssd import LLM, SamplingParams

    draft = args.ar_draft if backend == "ar" else args.block_draft

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

    print(f"=== {backend.upper()} RUN ===", flush=True)
    print(f"TARGET {args.target}", flush=True)
    print(f"DRAFT {draft}", flush=True)
    print(f"PROMPT_LEN {len(prompt_ids)}", flush=True)

    llm_kwargs = dict(
        draft=draft,
        speculate=True,
        draft_backend=backend,
        speculate_k=args.speculate_k,
        num_gpus=1,
        enforce_eager=True,
        max_num_seqs=1,
        max_model_len=args.max_model_len,
        kvcache_block_size=args.kvcache_block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        verbose=args.verbose_engine,
    )
    if backend == "block":
        llm_kwargs["block_draft_refine_steps"] = args.block_refine_steps
        llm_kwargs["block_draft_sampler"] = args.block_sampler
        llm_kwargs["block_draft_attention"] = args.block_attention
        llm_kwargs["block_draft_use_prefix_cache"] = args.block_prefix_cache
        llm_kwargs["block_draft_block_size"] = args.block_draft_block_size
        llm_kwargs["block_draft_mask_token_id"] = args.block_mask_token_id
        llm_kwargs["block_draft_special_tokens"] = args.block_special_tokens
        llm_kwargs["block_draft_forbid_token_ids"] = args.block_forbid_token_ids
        llm_kwargs["block_reuse_step_buffers"] = not args.block_no_reuse_step_buffers
        llm_kwargs["block_warm_start_mode"] = args.block_warm_start_mode
        llm_kwargs["block_warm_start_tokens"] = args.block_warm_start_tokens
        llm_kwargs["block_warm_start_draft"] = args.block_warm_start_draft
        llm_kwargs["block_warm_start_keep_loaded"] = args.block_warm_start_keep_loaded

    llm = LLM(args.target, **llm_kwargs)
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

    output = outputs[0]
    accepted_suffix = metrics.get("accepted_suffix_lens_with_recovery", [])
    result = {
        "backend": backend,
        "target": args.target,
        "draft": draft,
        "prompt_len": len(prompt_ids),
        "token_ids": output["token_ids"],
        "text": output["text"],
        "accepted_suffix_lens_with_recovery": accepted_suffix,
        "avg_tokens_per_step_incl_recovery": (
            sum(accepted_suffix) / len(accepted_suffix) if accepted_suffix else None
        ),
        "avg_accepted_speculative_fraction": _accepted_fraction(
            accepted_suffix, args.speculate_k
        ),
    }
    print(RESULT_PREFIX + json.dumps(result), flush=True)
    return result


def _child_args_from_parent(args: argparse.Namespace, mode: str) -> list[str]:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--mode", mode,
        "--target", args.target,
        "--ar-draft", args.ar_draft,
        "--block-draft", args.block_draft,
        "--system-prompt", args.system_prompt,
        "--user-prompt", args.user_prompt,
        "--max-new-tokens", str(args.max_new_tokens),
        "--speculate-k", str(args.speculate_k),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--kvcache-block-size", str(args.kvcache_block_size),
        "--block-refine-steps", str(args.block_refine_steps),
        "--block-sampler", args.block_sampler,
        "--block-attention", args.block_attention,
        "--block-special-tokens", args.block_special_tokens,
    ]
    if args.block_prefix_cache:
        cmd.append("--block-prefix-cache")
    if args.block_draft_block_size is not None:
        cmd.extend(["--block-draft-block-size", str(args.block_draft_block_size)])
    if args.block_mask_token_id is not None:
        cmd.extend(["--block-mask-token-id", str(args.block_mask_token_id)])
    if args.block_forbid_token_ids:
        cmd.extend(
            ["--block-forbid-token-ids"] +
            [str(token_id) for token_id in args.block_forbid_token_ids]
        )
    if args.block_no_reuse_step_buffers:
        cmd.append("--block-no-reuse-step-buffers")
    if args.block_warm_start_mode != "none":
        cmd.extend(["--block-warm-start-mode", args.block_warm_start_mode])
        cmd.extend(["--block-warm-start-tokens", str(args.block_warm_start_tokens)])
    if args.block_warm_start_draft is not None:
        cmd.extend(["--block-warm-start-draft", args.block_warm_start_draft])
    if args.block_warm_start_keep_loaded:
        cmd.append("--block-warm-start-keep-loaded")
    if args.verbose_engine:
        cmd.append("--verbose-engine")
    return cmd


def _run_child(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    cmd = _child_args_from_parent(args, mode)
    proc = subprocess.Popen(
        cmd,
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
    ar_result = _run_child(args, "ar")
    block_result = _run_child(args, "block")
    summary = {
        "target": args.target,
        "ar_draft": args.ar_draft,
        "block_draft": args.block_draft,
        "same_final_token_ids": ar_result["token_ids"] == block_result["token_ids"],
        "same_final_text": ar_result["text"] == block_result["text"],
        "ar": ar_result,
        "block": block_result,
    }
    print("=== COMPARISON SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.mode == "compare":
        _compare(args)
        return
    _run_single_backend(args, args.mode)


if __name__ == "__main__":
    main()
