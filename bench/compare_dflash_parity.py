"""Compare HF DFlash internals with native DFlash internals on one prompt.

The HF and native paths run in separate child processes to avoid loading two
Qwen3-8B copies at once. Each child saves CPU tensors, then the parent compares:

* target hidden feature used by DFlash
* first DFlash draft block tokens
* first DFlash draft block logits
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
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
    parser.add_argument("--mode", choices=["compare", "hf", "native"], default="compare")
    parser.add_argument("--target", type=str, default=_default_target())
    parser.add_argument("--dflash-draft", type=str, default=_default_dflash_draft())
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--user-prompt", type=str, default="Introduce yourself briefly.")
    parser.add_argument("--speculate-k", type=int, default=15)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--dflash-gpu-memory-reserve-gb", type=float, default=3.0)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def _prompt_ids(args: argparse.Namespace):
    from transformers import AutoTokenizer

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
    return tokenizer, prompt_ids


def _run_hf(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, DynamicCache

    tokenizer, prompt_ids = _prompt_ids(args)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    draft = AutoModel.from_pretrained(
        args.dflash_draft,
        trust_remote_code=True,
        dtype="auto",
    ).to("cuda").eval()
    target = AutoModelForCausalLM.from_pretrained(
        args.target,
        dtype="auto",
    ).to("cuda").eval()

    position_ids = torch.arange(
        input_ids.shape[1] + draft.block_size,
        device="cuda",
    ).unsqueeze(0)
    target_cache = DynamicCache()
    output = target(
        input_ids,
        position_ids=position_ids[:, :input_ids.shape[1]],
        past_key_values=target_cache,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    recovery = torch.argmax(output.logits, dim=-1)
    target_hidden = torch.cat(
        [output.hidden_states[layer_id + 1] for layer_id in draft.target_layer_ids],
        dim=-1,
    )
    block_output_ids = torch.full(
        (1, draft.block_size),
        draft.mask_token_id,
        dtype=torch.long,
        device="cuda",
    )
    block_output_ids[:, 0] = recovery[:, 0]
    noise_embedding = target.model.embed_tokens(block_output_ids)
    draft_cache = DynamicCache()
    draft_hidden = draft(
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=position_ids[:, :input_ids.shape[1] + draft.block_size],
        past_key_values=draft_cache,
        use_cache=True,
        is_causal=False,
    )
    logits = target.lm_head(draft_hidden[:, -draft.block_size + 1:, :])
    draft_tokens = torch.argmax(logits, dim=-1)

    data = {
        "prompt_len": len(prompt_ids),
        "target_layer_ids": list(draft.target_layer_ids),
        "recovery_token": int(recovery[0, 0].item()),
        "target_hidden": target_hidden[0].float().cpu(),
        "draft_tokens": draft_tokens[0].cpu(),
        "draft_logits": logits[0].float().cpu(),
        "draft_text": tokenizer.decode(draft_tokens[0].tolist()),
    }
    torch.save(data, args.out)
    result = {
        "mode": "hf",
        "out": args.out,
        "prompt_len": data["prompt_len"],
        "recovery_token": data["recovery_token"],
        "draft_tokens": data["draft_tokens"].tolist(),
        "draft_text": data["draft_text"],
    }
    print(RESULT_PREFIX + json.dumps(result), flush=True)
    return result


def _run_native(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from ssd import LLM, SamplingParams
    from ssd.engine.helpers.speculate_types import VerifyResult

    tokenizer, prompt_ids = _prompt_ids(args)
    llm = LLM(
        args.target,
        draft=args.dflash_draft,
        speculate=True,
        draft_backend="dflash",
        speculate_k=args.speculate_k,
        num_gpus=1,
        enforce_eager=True,
        max_num_seqs=1,
        max_model_len=args.max_model_len,
        kvcache_block_size=args.kvcache_block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dflash_gpu_memory_reserve_gb=args.dflash_gpu_memory_reserve_gb,
    )
    data = None
    try:
        llm.add_request(
            prompt_ids,
            SamplingParams(
                temperature=0.0,
                draft_temperature=0.0,
                max_new_tokens=args.speculate_k + 1,
                ignore_eos=True,
            ),
        )
        step = llm.create_inference_step(llm.config)
        seqs, is_prefill = llm.scheduler.schedule()
        if not is_prefill:
            raise RuntimeError("Expected native parity diagnostic to schedule prefill")
        verify_result = step.verifier.prefill(seqs, eagle=False)
        step.speculator.prefill(seqs, verify_result)
        speculate_result = step.speculator.speculate(
            seqs,
            VerifyResult([], [], None),
        )
        target_hidden = verify_result.dflash_acts
        draft_tokens = speculate_result.speculations[:, 1:][0]
        logits = speculate_result.logits_q[0]
        data = {
            "prompt_len": len(prompt_ids),
            "target_layer_ids": list(llm.config.dflash_target_layer_ids),
            "recovery_token": int(verify_result.recovery_tokens[0]),
            "target_hidden": target_hidden.float().cpu(),
            "draft_tokens": draft_tokens.cpu(),
            "draft_logits": logits.float().cpu(),
            "draft_text": tokenizer.decode(draft_tokens.tolist()),
        }
    finally:
        llm.exit(hard=False)
    if data is None:
        raise RuntimeError("Native parity run failed before collecting tensors")
    torch.save(data, args.out)
    result = {
        "mode": "native",
        "out": args.out,
        "prompt_len": data["prompt_len"],
        "recovery_token": data["recovery_token"],
        "draft_tokens": data["draft_tokens"].tolist(),
        "draft_text": data["draft_text"],
    }
    print(RESULT_PREFIX + json.dumps(result), flush=True)
    return result


def _child_cmd(args: argparse.Namespace, mode: str, out_path: str) -> list[str]:
    return [
        sys.executable,
        "-O",
        os.path.abspath(__file__),
        "--mode", mode,
        "--target", args.target,
        "--dflash-draft", args.dflash_draft,
        "--system-prompt", args.system_prompt,
        "--user-prompt", args.user_prompt,
        "--speculate-k", str(args.speculate_k),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--dflash-gpu-memory-reserve-gb", str(args.dflash_gpu_memory_reserve_gb),
        "--max-model-len", str(args.max_model_len),
        "--kvcache-block-size", str(args.kvcache_block_size),
        "--out", out_path,
    ]


def _run_child(cmd: list[str]) -> dict[str, Any]:
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
        raise RuntimeError(f"child run failed with exit code {returncode}")
    result_line = next(
        (line for line in reversed(lines) if line.startswith(RESULT_PREFIX)),
        None,
    )
    if result_line is None:
        raise RuntimeError("child run did not emit a JSON result")
    return json.loads(result_line[len(RESULT_PREFIX):])


def _tensor_stats(name: str, lhs, rhs) -> dict[str, Any]:
    diff = (lhs - rhs).abs()
    return {
        f"{name}_shape_lhs": list(lhs.shape),
        f"{name}_shape_rhs": list(rhs.shape),
        f"{name}_max_abs": float(diff.max().item()),
        f"{name}_mean_abs": float(diff.mean().item()),
    }


def _compare(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    with tempfile.TemporaryDirectory(prefix="dflash-parity-") as work_dir:
        hf_path = os.path.join(work_dir, "hf.pt")
        native_path = os.path.join(work_dir, "native.pt")
        hf_result = _run_child(_child_cmd(args, "hf", hf_path))
        native_result = _run_child(_child_cmd(args, "native", native_path))
        hf = torch.load(hf_path, map_location="cpu")
        native = torch.load(native_path, map_location="cpu")

    summary = {
        "same_recovery_token": hf["recovery_token"] == native["recovery_token"],
        "same_draft_tokens": hf["draft_tokens"].tolist() == native["draft_tokens"].tolist(),
        "hf": hf_result,
        "native": native_result,
    }
    summary.update(_tensor_stats("target_hidden", hf["target_hidden"], native["target_hidden"]))
    summary.update(_tensor_stats("draft_logits", hf["draft_logits"], native["draft_logits"]))
    print("=== DFLASH PARITY SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.mode == "hf":
        if args.out is None:
            raise ValueError("--out is required for --mode hf")
        _run_hf(args)
    elif args.mode == "native":
        if args.out is None:
            raise ValueError("--out is required for --mode native")
        _run_native(args)
    else:
        _compare(args)


if __name__ == "__main__":
    main()
