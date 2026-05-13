"""Compare upstream HF DFlash with this repo's native sync DFlash path."""

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
    parser.add_argument(
        "--mode",
        choices=["compare", "upstream", "native"],
        default="compare",
    )
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
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def _sample(logits, temperature: float = 0.0):
    import torch

    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    batch, seq_len, vocab = logits.shape
    probs = torch.softmax((logits / temperature).view(-1, vocab), dim=-1)
    return torch.multinomial(probs, num_samples=1).view(batch, seq_len)


def _extract_context_feature(hidden_states, layer_ids):
    return torch.cat([hidden_states[layer_id + 1] for layer_id in layer_ids], dim=-1)


def _run_upstream(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, DynamicCache

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

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + args.max_new_tokens
    block_size = draft.block_size
    mask_token_id = draft.mask_token_id

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device="cuda",
    )
    position_ids = torch.arange(output_ids.shape[1], device="cuda").unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    acceptance_lengths: list[int] = []

    torch.cuda.synchronize()
    start_time = perf_counter()

    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens + 1] = _sample(output.logits, 0.0)
    target_hidden = _extract_context_feature(
        output.hidden_states,
        draft.target_layer_ids,
    )

    start = num_input_tokens
    while start < max_length:
        block_output_ids = output_ids[:, start:start + block_size].clone()
        block_position_ids = position_ids[:, start:start + block_size]
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_hidden = draft(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[
                :,
                past_key_values_draft.get_seq_length():start + block_size,
            ],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )
        draft_logits = target.lm_head(draft_hidden[:, -block_size + 1:, :])
        past_key_values_draft.crop(start)
        block_output_ids[:, 1:] = _sample(draft_logits, 0.0)

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = _sample(output.logits, 0.0)
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1])
            .cumprod(dim=1)
            .sum(dim=1)[0]
            .item()
        )
        output_ids[:, start:start + acceptance_length + 1] = block_output_ids[
            :, :acceptance_length + 1
        ]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        target_hidden = _extract_context_feature(
            output.hidden_states,
            draft.target_layer_ids,
        )[:, :acceptance_length + 1, :]
        acceptance_lengths.append(acceptance_length + 1)

    torch.cuda.synchronize()
    total_time = perf_counter() - start_time

    completion_ids = output_ids[:, num_input_tokens:max_length][0].tolist()
    result = {
        "mode": "upstream",
        "target": args.target,
        "dflash_draft": args.dflash_draft,
        "prompt_len": len(prompt_ids),
        "decode_tokens": len(completion_ids),
        "total_time_s": total_time,
        "end_to_end_tok_s": len(completion_ids) / total_time if total_time > 0 else None,
        "accepted_suffix_lens_with_recovery": acceptance_lengths,
        "avg_tokens_per_step_incl_recovery": (
            sum(acceptance_lengths) / len(acceptance_lengths)
            if acceptance_lengths else None
        ),
        "token_ids": completion_ids,
        "text": tokenizer.decode(completion_ids),
    }
    print(RESULT_PREFIX + json.dumps(result), flush=True)
    return result


def _native_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-O",
        os.path.join(THIS_DIR, "compare_dflash_native.py"),
        "--mode", "dflash",
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
    return cmd


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


def _run_native(args: argparse.Namespace) -> dict[str, Any]:
    return _run_child(_native_cmd(args))


def _run_compare(args: argparse.Namespace) -> dict[str, Any]:
    upstream = _run_child([
        sys.executable,
        os.path.abspath(__file__),
        "--mode", "upstream",
        "--target", args.target,
        "--dflash-draft", args.dflash_draft,
        "--system-prompt", args.system_prompt,
        "--user-prompt", args.user_prompt,
        "--max-new-tokens", str(args.max_new_tokens),
        "--speculate-k", str(args.speculate_k),
    ])
    native = _run_native(args)
    summary = {
        "same_final_token_ids": upstream["token_ids"] == native["token_ids"],
        "same_final_text": upstream["text"] == native["text"],
        "native_vs_upstream_speed": (
            native["end_to_end_tok_s"] / upstream["end_to_end_tok_s"]
            if native["end_to_end_tok_s"] and upstream["end_to_end_tok_s"]
            else None
        ),
        "upstream": upstream,
        "native": native,
    }
    print("=== UPSTREAM COMPARISON SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.mode == "upstream":
        _run_upstream(args)
    elif args.mode == "native":
        _run_native(args)
    else:
        _run_compare(args)


if __name__ == "__main__":
    import torch

    main()
