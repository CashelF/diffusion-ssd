import gc
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from glob import glob
from typing import TYPE_CHECKING, Callable

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoModel, AutoTokenizer
try:
    from transformers import DynamicCache
except ImportError:
    from transformers.cache_utils import DynamicCache

from ssd.config import Config
from ssd.engine.helpers.speculate_types import VerifyResult
from ssd.engine.sequence import Sequence
from ssd.hf_remote import load_masked_lm

if TYPE_CHECKING:
    from ssd.engine.model_runner import ModelRunner


def _torch_dtype_from_config(config, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    torch_dtype = getattr(config, "torch_dtype", None)
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if isinstance(torch_dtype, str):
        name = torch_dtype.removeprefix("torch.")
        if hasattr(torch, name):
            return getattr(torch, name)
    return default


def _load_weight_from_model_path(model_path: str, weight_names: list[str]) -> torch.Tensor:
    safetensor_files = glob(os.path.join(model_path, "*.safetensors"))
    for file in safetensor_files:
        with safe_open(file, "pt", "cpu") as f:
            keys = set(f.keys())
            for weight_name in weight_names:
                if weight_name in keys:
                    return f.get_tensor(weight_name)

    bin_files = glob(os.path.join(model_path, "pytorch_model*.bin"))
    for file in bin_files:
        state_dict = torch.load(file, map_location="cpu")
        for weight_name in weight_names:
            if weight_name in state_dict:
                return state_dict[weight_name]

    raise FileNotFoundError(
        f"Could not find any of {weight_names} in model path {model_path}"
    )


def load_dflash_target_embed_and_lm_head(
    target_model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Embedding, nn.Linear]:
    """Load only target embedding and LM head weights for DFlash on draft rank."""
    embed_weight = _load_weight_from_model_path(
        target_model_path,
        ["model.embed_tokens.weight", "embed_tokens.weight"],
    )
    try:
        lm_head_weight = _load_weight_from_model_path(
            target_model_path,
            ["lm_head.weight"],
        )
    except FileNotFoundError:
        lm_head_weight = embed_weight

    vocab_size, hidden_size = embed_weight.shape
    embed_tokens = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
    embed_tokens.weight.data.copy_(embed_weight.to(device=device, dtype=dtype))

    lm_head = nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)
    if lm_head_weight.data_ptr() == embed_weight.data_ptr():
        lm_head.weight = embed_tokens.weight
    else:
        lm_head.weight.data.copy_(lm_head_weight.to(device=device, dtype=dtype))

    del embed_weight
    del lm_head_weight
    gc.collect()
    return embed_tokens.eval(), lm_head.eval()


class DFlashSeedGenerator:
    def __init__(
        self,
        config: Config,
        *,
        device: torch.device,
        target_embed_tokens: nn.Module,
        target_lm_head: nn.Module,
        dflash_model_path: str,
        dflash_hf_config,
    ):
        self.config = config
        self.device = device
        self.verbose = config.verbose
        self.block_size = config.dflash_block_size
        self.lookahead = config.speculate_k
        self.mask_token_id = config.dflash_mask_token_id
        self.target_layer_ids = config.dflash_target_layer_ids
        self.target_embed_tokens = target_embed_tokens
        self.target_lm_head = target_lm_head

        if self.block_size is None or self.target_layer_ids is None:
            raise ValueError("DFlash seed generator was initialized without resolved config")
        if self.mask_token_id is None:
            raise ValueError(
                "DFlash requires mask_token_id in the draft config or "
                "dflash_mask_token_id to be set explicitly"
            )

        torch_dtype = getattr(dflash_hf_config, "torch_dtype", None)
        model_kwargs = {"trust_remote_code": True}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        else:
            model_kwargs["dtype"] = "auto"
        default_device = torch.get_default_device()
        try:
            torch.set_default_device("cpu")
            self.model = AutoModel.from_pretrained(dflash_model_path, **model_kwargs)
        finally:
            torch.set_default_device(default_device)
        self.model.to(self.device)
        self.model.eval()

        self.past_key_values = DynamicCache()
        self.past_key_values_by_seq: dict[int, DynamicCache] = {}
        self.latest_target_hidden: torch.Tensor | None = None
        self.seed_calls = 0

    def reset(self) -> None:
        self.past_key_values = DynamicCache()
        self.past_key_values_by_seq = {}
        self.latest_target_hidden = None
        self.seed_calls = 0

    def _cache_for_seq(self, seq_id: int | None) -> DynamicCache:
        if seq_id is None:
            return self.past_key_values
        if seq_id not in self.past_key_values_by_seq:
            self.past_key_values_by_seq[seq_id] = DynamicCache()
        return self.past_key_values_by_seq[seq_id]

    def get_cache_len(self, seq_id: int | None = None) -> int:
        return int(self._cache_for_seq(seq_id).get_seq_length())

    def set_latest_target_hidden(self, target_hidden: torch.Tensor) -> None:
        target_hidden = target_hidden.unsqueeze(0) if target_hidden.dim() == 2 else target_hidden
        self.latest_target_hidden = target_hidden.to(self.device)

    def _target_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        try:
            return self.target_lm_head(hidden_states, last_only=False)
        except TypeError:
            return self.target_lm_head(hidden_states)

    @torch.inference_mode()
    def draft_from_token(
        self,
        *,
        recovery_token_id: int,
        start: int,
        lookahead: int,
        seq_id: int | None = None,
        prior_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run DFlash to produce a length-`lookahead` draft block.

        If `prior_tokens` is provided (V2 universal-drafter hit path), the
        masked block is initialized with those tokens instead of all
        mask tokens. The first position is still forced to
        `recovery_token_id` (the verified head). The block is then refined
        in a single forward pass conditioned on target hidden states.
        """
        if lookahead != self.lookahead or lookahead != self.block_size - 1:
            raise ValueError(
                "DFlash v1 requires lookahead == dflash_block_size - 1"
            )
        if self.latest_target_hidden is None:
            raise RuntimeError("DFlash draft called before target activations exist")

        past_key_values = self._cache_for_seq(seq_id)
        cache_len = int(past_key_values.get_seq_length())
        expected_target_len = start - cache_len
        if self.latest_target_hidden.shape[1] != expected_target_len:
            raise RuntimeError(
                "DFlash target activation backlog is out of sync with its cache: "
                f"seq_id={seq_id}, "
                f"cache_len={cache_len}, start={start}, "
                f"target_hidden_len={self.latest_target_hidden.shape[1]}"
            )

        block_output_ids = torch.full(
            (1, self.block_size),
            self.mask_token_id,
            dtype=torch.int64,
            device=self.device,
        )
        block_output_ids[0, 0] = recovery_token_id
        if prior_tokens is not None:
            prior_tokens = prior_tokens.to(self.device).to(torch.int64).flatten()
            # The prior is the tokens to draft (positions 1..block_size-1).
            # Drop anything beyond the block; if shorter, leave remaining
            # positions as mask tokens.
            n_prior = min(prior_tokens.numel(), self.block_size - 1)
            if n_prior > 0:
                block_output_ids[0, 1:1 + n_prior] = prior_tokens[:n_prior]
        position_ids = torch.arange(
            cache_len,
            start + self.block_size,
            dtype=torch.int64,
            device=self.device,
        ).unsqueeze(0)
        noise_embedding = self.target_embed_tokens(block_output_ids)

        draft_hidden = self.model(
            target_hidden=self.latest_target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            is_causal=False,
        )
        draft_hidden = draft_hidden[:, 1 - self.block_size:, :]
        logits_q = self._target_logits(draft_hidden)
        draft_tokens = logits_q.argmax(dim=-1)

        past_key_values.crop(start)
        self.seed_calls += 1
        return draft_tokens, logits_q

    def close(self) -> None:
        self.latest_target_hidden = None
        self.past_key_values = DynamicCache()
        self.past_key_values_by_seq = {}
        if getattr(self, "model", None) is not None:
            try:
                del self.model
            finally:
                self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SyncDraftBackendBase(ABC):
    @abstractmethod
    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> None:
        pass

    @abstractmethod
    def draft(
        self,
        seqs: list[Sequence],
        lookahead: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def reset(self) -> None:
        pass

    def observe_verify(
        self,
        seqs: list[Sequence],
        verify_result: VerifyResult,
    ) -> None:
        pass

    def close(self) -> None:
        pass


class AutoregressiveDraftBackend(SyncDraftBackendBase):
    def __init__(
        self,
        draft_model_runner: "ModelRunner",
        *,
        owns_runner: bool = True,
    ):
        self.draft_model_runner = draft_model_runner
        self.device = draft_model_runner.device
        self.owns_runner = owns_runner

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> None:
        assert not verify_result.eagle_acts, (
            "Eagle is not currently supported for synchronous speculation"
        )
        self.draft_model_runner.call("run", seqs, True)

    def draft(
        self,
        seqs: list[Sequence],
        lookahead: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(seqs)
        draft_tokens = torch.zeros(
            batch_size,
            lookahead,
            dtype=torch.int64,
            device=self.device,
        )
        logits_q = []

        # Extra forward at the end ensures the final speculative token is also
        # materialized in the draft KV cache.
        for k in range(lookahead + 1):
            token_ids, step_logits_q = self.draft_model_runner.call(
                "run", seqs, False, True, True
            )
            for seq in seqs:
                seq.num_draft_cached_tokens += 1

            if k == lookahead:
                break

            logits_q.append(step_logits_q)
            token_ids_t = torch.tensor(
                token_ids,
                dtype=torch.int64,
                device=self.device,
            )
            draft_tokens[:, k] = token_ids_t

            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)

        return draft_tokens, torch.stack(logits_q, dim=1)

    def close(self) -> None:
        if self.owns_runner and self.draft_model_runner is not None:
            try:
                self.draft_model_runner.exit(hard=False)
            finally:
                self.draft_model_runner = None


class SwitchingDraftBackend(SyncDraftBackendBase):
    """Route sync drafting between a warm-start backend and a primary backend.

    This keeps the implementation usable on a single GPU by default: only the
    active backend is kept resident unless `keep_loaded=True`.
    """

    def __init__(
        self,
        *,
        primary_factory: Callable[[], SyncDraftBackendBase],
        primary_name: str,
        warm_factory: Callable[[], SyncDraftBackendBase] | None = None,
        warm_name: str | None = None,
        warm_start_tokens: int = 0,
        keep_loaded: bool = False,
        verbose: bool = False,
        initial_warm_backend: SyncDraftBackendBase | None = None,
    ):
        self.primary_factory = primary_factory
        self.primary_name = primary_name
        self.warm_factory = warm_factory
        self.warm_name = warm_name or "warm"
        self.warm_start_tokens = warm_start_tokens
        self.keep_loaded = keep_loaded
        self.verbose = verbose

        self._backends: dict[str, SyncDraftBackendBase | None] = {
            "primary": None,
            "warm": initial_warm_backend,
        }
        self._active_stage: str | None = None
        self._active_backend: SyncDraftBackendBase | None = None
        self.reset()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _get_or_create_backend(self, stage: str) -> SyncDraftBackendBase:
        backend = self._backends.get(stage)
        if backend is not None:
            return backend

        if stage == "primary":
            backend = self.primary_factory()
        elif stage == "warm" and self.warm_factory is not None:
            backend = self.warm_factory()
        else:
            raise RuntimeError(f"No backend factory available for stage={stage!r}")

        self._backends[stage] = backend
        return backend

    def _activate(self, stage: str) -> None:
        if self._active_stage == stage:
            return

        if not self.keep_loaded and self._active_stage is not None:
            old_backend = self._backends.get(self._active_stage)
            if old_backend is not None:
                old_backend.close()
                self._backends[self._active_stage] = None

        self._active_backend = self._get_or_create_backend(stage)
        self._active_stage = stage
        self._log(
            f"[draft_router] activated {self.primary_name if stage == 'primary' else self.warm_name}"
        )

    def _should_switch_to_primary(self, seqs: list[Sequence]) -> bool:
        if self.warm_factory is None or self.warm_start_tokens <= 0:
            return True
        return all(
            seq.num_completion_tokens >= self.warm_start_tokens
            for seq in seqs
        )

    def reset(self) -> None:
        if self.warm_factory is not None and self.warm_start_tokens > 0:
            self._activate("warm")
        else:
            self._activate("primary")

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> None:
        assert self._active_backend is not None
        self._active_backend.prefill(seqs, verify_result)

    def draft(
        self,
        seqs: list[Sequence],
        lookahead: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._active_stage != "primary" and self._should_switch_to_primary(seqs):
            self._activate("primary")
        assert self._active_backend is not None
        return self._active_backend.draft(seqs, lookahead)

    def observe_verify(
        self,
        seqs: list[Sequence],
        verify_result: VerifyResult,
    ) -> None:
        assert self._active_backend is not None
        self._active_backend.observe_verify(seqs, verify_result)

    def close(self) -> None:
        seen = set()
        for stage in ("warm", "primary"):
            backend = self._backends.get(stage)
            if backend is None or id(backend) in seen:
                continue
            seen.add(id(backend))
            backend.close()
            self._backends[stage] = None
        self._active_backend = None
        self._active_stage = None


class DFlashDraftBackend(SyncDraftBackendBase):
    def __init__(self, config: Config, target_model_runner: "ModelRunner"):
        self.config = config
        self.device = target_model_runner.device
        self.verbose = config.verbose
        self.block_size = config.dflash_block_size
        self.lookahead = config.speculate_k
        self.mask_token_id = config.dflash_mask_token_id
        self.target_layer_ids = config.dflash_target_layer_ids

        if self.block_size is None or self.target_layer_ids is None:
            raise ValueError("DFlash backend was initialized without resolved config")
        if self.mask_token_id is None:
            raise ValueError(
                "draft_backend='dflash' requires mask_token_id in the draft "
                "config or dflash_mask_token_id to be set explicitly"
            )

        self.seed_generator = DFlashSeedGenerator(
            config,
            device=self.device,
            target_embed_tokens=target_model_runner.model.model.embed_tokens,
            target_lm_head=target_model_runner.model.lm_head,
            dflash_model_path=config.draft,
            dflash_hf_config=config.dflash_hf_config or config.draft_hf_config,
        )

    def reset(self) -> None:
        self.seed_generator.reset()

    def _validate_sampling_mode(self, seqs: list[Sequence], lookahead: int) -> None:
        if len(seqs) != 1:
            raise NotImplementedError(
                "draft_backend='dflash' v1 supports batch size 1 only"
            )
        if lookahead != self.lookahead or lookahead != self.block_size - 1:
            raise ValueError(
                "draft_backend='dflash' v1 requires lookahead == "
                "dflash_block_size - 1"
            )
        for seq in seqs:
            draft_temp = (
                seq.draft_temperature
                if seq.draft_temperature is not None
                else seq.temperature
            )
            if seq.temperature != 0 or draft_temp != 0:
                raise NotImplementedError(
                    "draft_backend='dflash' currently supports greedy decoding "
                    "only (temperature=0, draft_temperature=0)."
                )

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> None:
        self._validate_sampling_mode(seqs, self.lookahead)
        if verify_result.dflash_acts is None:
            raise RuntimeError(
                "DFlash prefill requires target activations from the verifier"
            )
        acts = verify_result.dflash_acts
        self.seed_generator.reset()
        self.seed_generator.set_latest_target_hidden(acts)

    @torch.inference_mode()
    def draft(
        self,
        seqs: list[Sequence],
        lookahead: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_sampling_mode(seqs, lookahead)
        seq = seqs[0]
        start = len(seq.token_ids) - 1
        cache_len = self.seed_generator.get_cache_len()
        draft_tokens, logits_q = self.seed_generator.draft_from_token(
            recovery_token_id=seq.token_ids[-1],
            start=start,
            lookahead=lookahead,
        )

        tokens = draft_tokens[0].tolist()
        seq.token_ids.extend(tokens)
        seq.num_tokens += len(tokens)
        seq.last_token = seq.token_ids[-1]

        if self.verbose:
            print(
                f"[dflash_draft] generated {lookahead} draft tokens "
                f"(block_size={self.block_size}, cache_len={cache_len}, start={start})",
                flush=True,
            )

        return draft_tokens, logits_q

    def observe_verify(
        self,
        seqs: list[Sequence],
        verify_result: VerifyResult,
    ) -> None:
        if verify_result.dflash_acts is None:
            raise RuntimeError(
                "DFlash observe_verify requires target activations from verifier"
            )
        if len(seqs) != 1:
            raise NotImplementedError(
                "draft_backend='dflash' v1 supports batch size 1 only"
            )
        accepted_len = len(verify_result.new_suffixes[0])
        self.seed_generator.set_latest_target_hidden(verify_result.dflash_acts[
            :1,
            :accepted_len,
            :,
        ].clone())

    def close(self) -> None:
        self.seed_generator.close()


@dataclass
class _BlockPrefixCacheState:
    prefix_tokens: list[int] = field(default_factory=list)
    past_key_values: object | None = None


class BlockDiffusionDraftBackend(SyncDraftBackendBase):
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.verbose = config.verbose
        self.refine_steps = config.block_draft_refine_steps
        self.sampler = config.block_draft_sampler
        self.attention_mode = config.block_draft_attention
        self.use_prefix_cache = config.block_draft_use_prefix_cache
        self.block_size = config.block_draft_block_size or config.speculate_k
        self.special_token_mode = config.block_draft_special_tokens
        self.reuse_step_buffers = config.block_reuse_step_buffers
        self.target_vocab_size = config.hf_config.vocab_size
        self.forbid_token_ids = sorted(
            set(config.block_draft_forbid_token_ids or [])
        )

        # We intentionally tokenize with the target tokenizer so token ids fed
        # into verification always live in the verifier's vocabulary.
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        draft_tokenizer = AutoTokenizer.from_pretrained(config.draft, use_fast=True)
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = draft_tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        if self.pad_token_id is None:
            self.pad_token_id = 0

        self.mask_token_id = config.block_draft_mask_token_id
        if self.mask_token_id is None:
            self.mask_token_id = self.tokenizer.mask_token_id
        if self.mask_token_id is None:
            self.mask_token_id = draft_tokenizer.mask_token_id
        if self.mask_token_id is None:
            raise ValueError(
                "draft_backend='block' requires a tokenizer with mask_token_id, "
                "or block_draft_mask_token_id must be set explicitly."
            )

        special_ids = set(self.tokenizer.all_special_ids or [])
        special_ids.update(draft_tokenizer.all_special_ids or [])
        self.special_token_ids = sorted(special_ids)

        torch_dtype = getattr(config.draft_hf_config, "torch_dtype", None)
        self.model = load_masked_lm(
            config.draft,
            torch_dtype=torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.use_boolean_attention = (
            getattr(self.model.config, "_attn_implementation", None) != "eager"
        )

        if self.model.config.vocab_size != self.target_vocab_size:
            raise ValueError(
                "Block draft model vocab size must match target vocab size. "
                f"Got draft={self.model.config.vocab_size}, "
                f"target={self.target_vocab_size}."
            )

        self._input_ids_buf = None
        self._attention_mask_buf = None
        self._offset_cache: dict[int, torch.Tensor] = {}
        self._prefix_cache_states: dict[int, _BlockPrefixCacheState] = {}

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> None:
        if verify_result.eagle_acts is not None:
            raise NotImplementedError(
                "draft_backend='block' does not currently support EAGLE conditioning"
            )
        if self.use_prefix_cache and verify_result.recovery_tokens:
            for seq, recovery_token in zip(seqs, verify_result.recovery_tokens):
                prefix_tokens = list(seq.token_ids)
                if recovery_token is not None:
                    prefix_tokens.append(recovery_token)
                self._ensure_prefix_cache(seq.seq_id, prefix_tokens)

    def reset(self) -> None:
        self._prefix_cache_states.clear()

    def _validate_sampling_mode(self, seqs: list[Sequence]) -> None:
        for seq in seqs:
            draft_temp = (
                seq.draft_temperature
                if seq.draft_temperature is not None
                else seq.temperature
            )
            if seq.temperature != 0 or draft_temp != 0:
                raise NotImplementedError(
                    "draft_backend='block' currently supports greedy decoding "
                    "only (temperature=0, draft_temperature=0)."
                )

    def _ensure_step_buffers(self, batch_size: int, max_total_len: int) -> None:
        if (
            self._input_ids_buf is not None
            and self._input_ids_buf.shape[0] >= batch_size
            and self._input_ids_buf.shape[1] >= max_total_len
        ):
            return

        self._input_ids_buf = torch.empty(
            batch_size,
            max_total_len,
            dtype=torch.int64,
            device=self.device,
        )
        self._attention_mask_buf = torch.empty(
            batch_size,
            max_total_len,
            dtype=torch.int64,
            device=self.device,
        )

    def _get_offsets(self, lookahead: int) -> torch.Tensor:
        offsets = self._offset_cache.get(lookahead)
        if offsets is None:
            offsets = torch.arange(lookahead, dtype=torch.int64, device=self.device)
            self._offset_cache[lookahead] = offsets
        return offsets

    @staticmethod
    def _common_prefix_len(a: list[int], b: list[int]) -> int:
        limit = min(len(a), len(b))
        i = 0
        while i < limit and a[i] == b[i]:
            i += 1
        return i

    def _build_staircase_attention_mask(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        batch_size, seq_len = input_ids.shape
        valid = input_ids != self.pad_token_id
        pos_raw = torch.cumsum(valid.long(), dim=-1)
        position_ids = torch.where(
            valid,
            pos_raw - 1,
            torch.zeros_like(pos_raw),
        ).long()

        col = torch.arange(seq_len, device=input_ids.device)
        block_ids = (col // self.block_size).view(1, seq_len).expand(batch_size, seq_len)
        block_ids = torch.where(
            valid,
            block_ids,
            torch.full_like(block_ids, -1),
        )

        q = block_ids.view(batch_size, 1, seq_len, 1)
        k = block_ids.view(batch_size, 1, 1, seq_len)
        allowed = (k <= q) & (q >= 0) & (k >= 0)

        if self.use_boolean_attention:
            attention_mask = allowed
        else:
            attention_mask = torch.zeros(
                batch_size,
                1,
                seq_len,
                seq_len,
                dtype=torch.float32,
                device=input_ids.device,
            )
            attention_mask.masked_fill_(~allowed, torch.finfo(torch.float32).min)

        return attention_mask, position_ids

    @staticmethod
    def _transfer_schedule(num_masked: int, steps: int) -> list[int]:
        base = num_masked // steps
        rem = num_masked % steps
        return [base + (1 if i < rem else 0) for i in range(steps)]

    def _prepare_prefix_inputs(
        self,
        prefixes: list[list[int]],
        lookahead: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(prefixes)
        prefix_lens = [len(prefix) for prefix in prefixes]
        max_total_len = max(prefix_len + lookahead for prefix_len in prefix_lens)

        if self.reuse_step_buffers:
            self._ensure_step_buffers(batch_size, max_total_len)
            input_ids = self._input_ids_buf[:batch_size, :max_total_len]
            attention_mask = self._attention_mask_buf[:batch_size, :max_total_len]
        else:
            input_ids = torch.empty(
                batch_size,
                max_total_len,
                dtype=torch.int64,
                device=self.device,
            )
            attention_mask = torch.empty(
                batch_size,
                max_total_len,
                dtype=torch.int64,
                device=self.device,
            )

        input_ids.fill_(self.pad_token_id)
        attention_mask.zero_()

        prefix_lens_t = torch.tensor(
            prefix_lens,
            dtype=torch.int64,
            device=self.device,
        )

        for i, prefix in enumerate(prefixes):
            prefix_tensor = torch.tensor(prefix, dtype=torch.int64, device=self.device)
            prefix_len = prefix_tensor.shape[0]
            input_ids[i, :prefix_len] = prefix_tensor
            attention_mask[i, :prefix_len + lookahead] = 1

        return input_ids, attention_mask, prefix_lens_t

    def _write_block_tokens(
        self,
        input_ids: torch.Tensor,
        prefix_lens: torch.Tensor,
        block_tokens: torch.Tensor,
        lookahead: int,
    ) -> None:
        for i, prefix_len in enumerate(prefix_lens.tolist()):
            input_ids[i, prefix_len:prefix_len + lookahead] = block_tokens[i]

    def _select_block_logits(
        self,
        logits: torch.Tensor,
        prefix_lens: torch.Tensor,
        lookahead: int,
    ) -> torch.Tensor:
        batch_idx = torch.arange(logits.shape[0], device=self.device)[:, None]
        pos_idx = prefix_lens[:, None] + self._get_offsets(lookahead)[None, :]
        return logits[batch_idx, pos_idx]

    def _apply_token_constraints(
        self,
        block_logits: torch.Tensor,
    ) -> torch.Tensor:
        need_special_filter = (
            self.special_token_mode != "none" and len(self.special_token_ids) > 0
        )
        need_forbid_filter = len(self.forbid_token_ids) > 0
        if not need_special_filter and not need_forbid_filter:
            return block_logits

        block_logits = block_logits.clone()

        if need_special_filter:
            if self.special_token_mode == "all":
                block_logits[:, :, self.special_token_ids] = float("-inf")
            elif block_logits.shape[1] > 1:
                block_logits[:, :-1, self.special_token_ids] = float("-inf")

        if need_forbid_filter:
            block_logits[:, :, self.forbid_token_ids] = float("-inf")

        return block_logits

    def _ensure_prefix_cache(
        self,
        seq_id: int,
        prefix_tokens: list[int],
    ) -> _BlockPrefixCacheState:
        state = self._prefix_cache_states.setdefault(seq_id, _BlockPrefixCacheState())

        if not prefix_tokens:
            state.prefix_tokens = []
            state.past_key_values = None
            return state

        lcp = self._common_prefix_len(state.prefix_tokens, prefix_tokens)
        if lcp < len(state.prefix_tokens):
            if state.past_key_values is not None:
                if lcp > 0:
                    state.past_key_values.crop(lcp)
                else:
                    state.past_key_values = None
            state.prefix_tokens = state.prefix_tokens[:lcp]

        if state.past_key_values is not None and len(state.prefix_tokens) == len(prefix_tokens):
            return state

        prefix_tensor = torch.tensor(
            prefix_tokens,
            dtype=torch.int64,
            device=self.device,
        ).unsqueeze(0)
        attention_mask, position_ids = self._build_staircase_attention_mask(prefix_tensor)

        if state.past_key_values is None or lcp == 0:
            outputs = self.model(
                prefix_tensor,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
            state.past_key_values = outputs.past_key_values
        elif lcp < len(prefix_tokens):
            extension = prefix_tensor[:, lcp:]
            extension_attention = attention_mask[:, :, lcp:, :]
            extension_positions = position_ids[:, lcp:]
            outputs = self.model(
                extension,
                attention_mask=extension_attention,
                position_ids=extension_positions,
                past_key_values=state.past_key_values,
                use_cache=True,
            )
            if outputs.past_key_values is not None:
                state.past_key_values = outputs.past_key_values

        state.prefix_tokens = prefix_tokens.copy()
        return state

    def _prepare_staircase_context(
        self,
        prefix_tokens: list[int],
        lookahead: int,
        seq_id: int,
    ) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, _BlockPrefixCacheState | None]:
        prefix_len = len(prefix_tokens)
        total_tokens = torch.full(
            (1, prefix_len + lookahead),
            self.mask_token_id,
            dtype=torch.int64,
            device=self.device,
        )
        if prefix_len > 0:
            prefix_tensor = torch.tensor(
                prefix_tokens,
                dtype=torch.int64,
                device=self.device,
            )
            total_tokens[0, :prefix_len] = prefix_tensor
        else:
            prefix_tensor = torch.empty(0, dtype=torch.int64, device=self.device)

        full_attention_mask, full_position_ids = self._build_staircase_attention_mask(
            total_tokens
        )
        prefix_state = None
        if self.use_prefix_cache and prefix_len > 0:
            prefix_state = self._ensure_prefix_cache(seq_id, prefix_tokens)

        return (
            prefix_tensor,
            prefix_len,
            full_attention_mask,
            full_position_ids,
            prefix_state,
        )

    def _forward_staircase_block(
        self,
        prefix_tensor: torch.Tensor,
        prefix_len: int,
        block_tokens: torch.Tensor,
        full_attention_mask: torch.Tensor,
        full_position_ids: torch.Tensor,
        prefix_state: _BlockPrefixCacheState | None,
    ) -> torch.Tensor:
        block_tokens = block_tokens.unsqueeze(0)

        if prefix_state is not None and prefix_state.past_key_values is not None:
            block_attention = full_attention_mask[:, :, prefix_len:, :]
            block_positions = full_position_ids[:, prefix_len:]
            past_key_values = prefix_state.past_key_values
            try:
                outputs = self.model(
                    block_tokens,
                    attention_mask=block_attention,
                    position_ids=block_positions,
                    past_key_values=past_key_values,
                    use_cache=False,
                )
            finally:
                past_key_values.crop(prefix_len)
            return outputs.logits[0]

        total_tokens = torch.empty(
            1,
            prefix_len + block_tokens.shape[1],
            dtype=torch.int64,
            device=self.device,
        )
        if prefix_len > 0:
            total_tokens[0, :prefix_len] = prefix_tensor
        total_tokens[0, prefix_len:] = block_tokens[0]

        outputs = self.model(
            total_tokens,
            attention_mask=full_attention_mask,
            position_ids=full_position_ids,
        )
        return outputs.logits[0, prefix_len:]

    def _draft_mask_predict_full(
        self,
        prefixes: list[list[int]],
        lookahead: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(prefixes)
        input_ids, attention_mask, prefix_lens = self._prepare_prefix_inputs(
            prefixes,
            lookahead,
        )
        block_tokens = torch.full(
            (batch_size, lookahead),
            self.mask_token_id,
            dtype=torch.int64,
            device=self.device,
        )
        locked = torch.zeros(
            (batch_size, lookahead),
            dtype=torch.bool,
            device=self.device,
        )
        final_logits = None

        for step in range(self.refine_steps):
            self._write_block_tokens(input_ids, prefix_lens, block_tokens, lookahead)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            block_logits = self._select_block_logits(
                outputs.logits,
                prefix_lens,
                lookahead,
            )
            block_logits = self._apply_token_constraints(block_logits)
            pred_ids = block_logits.argmax(dim=-1)
            final_logits = block_logits

            if step == self.refine_steps - 1:
                block_tokens = torch.where(locked, block_tokens, pred_ids)
                break

            next_tokens = block_tokens.clone()
            next_tokens[~locked] = pred_ids[~locked]
            target_locked = math.ceil(lookahead * (step + 1) / self.refine_steps)

            for i in range(batch_size):
                already_locked = int(locked[i].sum().item())
                if target_locked > already_locked:
                    need = min(
                        target_locked - already_locked,
                        lookahead - already_locked,
                    )
                    conf = block_logits[i].amax(dim=-1)
                    conf = conf.masked_fill(locked[i], float("-inf"))
                    topk = torch.topk(conf, k=need).indices
                    locked[i, topk] = True
                next_tokens[i, ~locked[i]] = self.mask_token_id

            block_tokens = next_tokens

        if final_logits is None:
            raise RuntimeError("Block draft backend did not produce logits")
        return block_tokens, final_logits

    def _draft_remask_full(
        self,
        prefixes: list[list[int]],
        lookahead: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, attention_mask, prefix_lens = self._prepare_prefix_inputs(
            prefixes,
            lookahead,
        )
        block_tokens = torch.full(
            (len(prefixes), lookahead),
            self.mask_token_id,
            dtype=torch.int64,
            device=self.device,
        )
        final_logits = None

        for step in range(self.refine_steps):
            self._write_block_tokens(input_ids, prefix_lens, block_tokens, lookahead)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            block_logits = self._select_block_logits(
                outputs.logits,
                prefix_lens,
                lookahead,
            )
            block_logits = self._apply_token_constraints(block_logits)
            pred_ids = block_logits.argmax(dim=-1)
            final_logits = block_logits

            if step == self.refine_steps - 1:
                block_tokens = pred_ids
                break

            num_keep = math.ceil(lookahead * (step + 1) / self.refine_steps)
            num_remask = max(0, lookahead - num_keep)
            next_tokens = pred_ids.clone()

            if num_remask > 0:
                conf = block_logits.amax(dim=-1)
                remask_idx = torch.topk(
                    conf,
                    k=num_remask,
                    dim=-1,
                    largest=False,
                ).indices
                next_tokens.scatter_(
                    1,
                    remask_idx,
                    torch.full_like(remask_idx, self.mask_token_id),
                )

            block_tokens = next_tokens

        if final_logits is None:
            raise RuntimeError("Block draft backend did not produce logits")
        return block_tokens, final_logits

    def _draft_staircase_single(
        self,
        prefix_tokens: list[int],
        lookahead: int,
        seq_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            prefix_tensor,
            prefix_len,
            full_attention_mask,
            full_position_ids,
            prefix_state,
        ) = self._prepare_staircase_context(prefix_tokens, lookahead, seq_id)

        block_tokens = torch.full(
            (lookahead,),
            self.mask_token_id,
            dtype=torch.int64,
            device=self.device,
        )
        locked = torch.zeros(
            lookahead,
            dtype=torch.bool,
            device=self.device,
        )
        transfer_schedule = self._transfer_schedule(lookahead, self.refine_steps)
        final_logits = None

        for step in range(self.refine_steps):
            block_logits = self._forward_staircase_block(
                prefix_tensor,
                prefix_len,
                block_tokens,
                full_attention_mask,
                full_position_ids,
                prefix_state,
            ).unsqueeze(0)
            block_logits = self._apply_token_constraints(block_logits)[0]
            pred_ids = block_logits.argmax(dim=-1)
            final_logits = block_logits

            if step == self.refine_steps - 1:
                if self.sampler == "remask":
                    block_tokens = pred_ids
                else:
                    block_tokens = torch.where(locked, block_tokens, pred_ids)
                break

            conf = block_logits.gather(-1, pred_ids.unsqueeze(-1)).squeeze(-1)

            if self.sampler == "remask":
                num_keep = math.ceil(lookahead * (step + 1) / self.refine_steps)
                num_remask = max(0, lookahead - num_keep)
                next_tokens = pred_ids.clone()
                if num_remask > 0:
                    remask_idx = torch.topk(
                        conf,
                        k=num_remask,
                        largest=False,
                    ).indices
                    next_tokens.scatter_(
                        0,
                        remask_idx,
                        torch.full_like(remask_idx, self.mask_token_id),
                    )
                block_tokens = next_tokens
                continue

            if self.sampler == "first_hitting":
                num_commit = transfer_schedule[step]
            else:
                already_locked = int(locked.sum().item())
                target_locked = math.ceil(lookahead * (step + 1) / self.refine_steps)
                num_commit = min(
                    max(0, target_locked - already_locked),
                    lookahead - already_locked,
                )

            if num_commit > 0:
                conf = conf.masked_fill(locked, float("-inf"))
                topk = torch.topk(conf, k=num_commit).indices
                locked[topk] = True
                block_tokens[topk] = pred_ids[topk]

            block_tokens[~locked] = self.mask_token_id

        if final_logits is None:
            raise RuntimeError("Block draft backend did not produce logits")
        return block_tokens, final_logits

    def _draft_staircase(
        self,
        seqs: list[Sequence],
        lookahead: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        draft_tokens = []
        logits_q = []
        for seq in seqs:
            seq_tokens, seq_logits = self._draft_staircase_single(
                list(seq.token_ids),
                lookahead,
                seq.seq_id,
            )
            draft_tokens.append(seq_tokens)
            logits_q.append(seq_logits)

        return torch.stack(draft_tokens, dim=0), torch.stack(logits_q, dim=0)

    @torch.inference_mode()
    def draft(
        self,
        seqs: list[Sequence],
        lookahead: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_sampling_mode(seqs)
        use_staircase = (
            self.attention_mode == "staircase"
            or self.use_prefix_cache
            or self.sampler == "first_hitting"
        )

        if use_staircase:
            block_tokens, final_logits = self._draft_staircase(seqs, lookahead)
        else:
            prefixes = [list(seq.token_ids) for seq in seqs]
            if self.sampler == "mask_predict":
                block_tokens, final_logits = self._draft_mask_predict_full(
                    prefixes,
                    lookahead,
                )
            elif self.sampler == "remask":
                block_tokens, final_logits = self._draft_remask_full(
                    prefixes,
                    lookahead,
                )
            else:
                raise ValueError(f"Unsupported block sampler: {self.sampler}")

        drafted = block_tokens.tolist()
        for seq, tokens in zip(seqs, drafted):
            seq.token_ids.extend(tokens)
            seq.num_tokens += len(tokens)
            seq.last_token = seq.token_ids[-1]

        if self.verbose:
            print(
                f"[block_draft] generated {lookahead} draft tokens in "
                f"{self.refine_steps} parallel refinement steps "
                f"(sampler={self.sampler}, attention={self.attention_mode}, "
                f"prefix_cache={self.use_prefix_cache}, block_size={self.block_size})",
                flush=True,
            )

        return block_tokens, final_logits

    def close(self) -> None:
        self._prefix_cache_states.clear()
        if getattr(self, "model", None) is not None:
            try:
                del self.model
            finally:
                self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
