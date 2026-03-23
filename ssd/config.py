import os
from dataclasses import dataclass
from transformers import AutoConfig
import torch
from ssd.paths import DEFAULT_TARGET, DEFAULT_DRAFT
from ssd.hf_remote import load_config

@dataclass
class Config:
    model: str = DEFAULT_TARGET
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 1 
    max_model_len: int = 4096 
    gpu_memory_utilization: float = 0.7
    num_gpus: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # spec config args
    draft_hf_config: AutoConfig | None = None
    speculate: bool = False 
    draft: str = DEFAULT_DRAFT
    draft_backend: str = "ar"
    speculate_k: int = 1
    draft_async: bool = False
    block_draft_refine_steps: int = 4
    block_draft_sampler: str = "mask_predict"
    block_draft_attention: str = "full"
    block_draft_use_prefix_cache: bool = False
    block_draft_block_size: int | None = None
    block_draft_mask_token_id: int | None = None
    block_draft_special_tokens: str = "none"
    block_draft_forbid_token_ids: list[int] | None = None
    block_reuse_step_buffers: bool = True
    block_warm_start_mode: str = "none"
    block_warm_start_tokens: int = 0
    block_warm_start_draft: str | None = None
    block_warm_start_keep_loaded: bool = False
    block_warm_start_draft_hf_config: AutoConfig | None = None
    
    # async spec only
    async_fan_out: int = 3
    fan_out_list: list[int] | None = None
    fan_out_list_miss: list[int] | None = None
    sampler_x: float | None = None 
    jit_speculate: bool = False 

    # eagle3
    use_eagle: bool = False 
    eagle_layers: list[int] | None = None   
    d_model_target: int | None = None
    tokenizer_path: str | None = None

    # Debugging
    verbose: bool = False 
    debug_mode: bool = False 
    max_steps: int | None = None

    @property
    def max_blocks(self): 
        return (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size

    def __post_init__(self):
        model = self.model 
        assert os.path.isdir(model)
        assert self.draft_backend in {"ar", "block"}, (
            f"Unsupported draft_backend={self.draft_backend!r}. "
            "Expected one of: 'ar', 'block'."
        )
        assert self.block_draft_sampler in {"mask_predict", "remask", "first_hitting"}, (
            f"Unsupported block_draft_sampler={self.block_draft_sampler!r}. "
            "Expected one of: 'mask_predict', 'remask', 'first_hitting'."
        )
        assert self.block_draft_attention in {"full", "staircase"}, (
            f"Unsupported block_draft_attention={self.block_draft_attention!r}. "
            "Expected one of: 'full', 'staircase'."
        )
        assert self.block_draft_refine_steps >= 1, (
            "block_draft_refine_steps must be >= 1"
        )
        if self.block_draft_block_size is not None:
            assert self.block_draft_block_size >= 1, (
                "block_draft_block_size must be >= 1 when set"
            )
        assert self.block_draft_special_tokens in {"none", "interior", "all"}, (
            f"Unsupported block_draft_special_tokens={self.block_draft_special_tokens!r}. "
            "Expected one of: 'none', 'interior', 'all'."
        )
        assert self.block_warm_start_mode in {"none", "ar"}, (
            f"Unsupported block_warm_start_mode={self.block_warm_start_mode!r}. "
            "Expected one of: 'none', 'ar'."
        )
        assert self.block_warm_start_tokens >= 0, (
            "block_warm_start_tokens must be >= 0"
        )
        if self.block_draft_forbid_token_ids is not None:
            self.block_draft_forbid_token_ids = [
                int(token_id) for token_id in self.block_draft_forbid_token_ids
            ]

        assert 1 <= self.num_gpus <= 8 # this codebase only works on one node 
        self.hf_config = AutoConfig.from_pretrained(model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings) 
        if self.speculate: 
            draft = self.draft
            self.draft_hf_config = load_config(
                draft,
                trust_remote_code=(self.draft_backend == "block"),
            )
            if self.draft_backend == "block" and self.block_draft_block_size is None:
                self.block_draft_block_size = self.speculate_k
            self.max_model_len = min(
                self.max_model_len, self.draft_hf_config.max_position_embeddings)
            if self.draft_backend == "block":
                assert not self.draft_async, (
                    "draft_backend='block' currently only supports synchronous "
                    "speculative decoding"
                )
                if self.block_draft_use_prefix_cache:
                    assert self.block_draft_attention == "staircase", (
                        "block_draft_use_prefix_cache requires "
                        "block_draft_attention='staircase'"
                    )
                if self.block_draft_sampler == "first_hitting":
                    assert self.block_draft_attention == "staircase", (
                        "block_draft_sampler='first_hitting' requires "
                        "block_draft_attention='staircase'"
                    )
                if self.block_warm_start_mode == "ar":
                    assert self.block_warm_start_tokens > 0, (
                        "block_warm_start_mode='ar' requires "
                        "block_warm_start_tokens > 0"
                    )
                    assert self.block_warm_start_draft is not None, (
                        "block_warm_start_mode='ar' requires "
                        "block_warm_start_draft to be set"
                    )
                    assert os.path.isdir(self.block_warm_start_draft), (
                        f"Warm-start draft path does not exist: "
                        f"{self.block_warm_start_draft}"
                    )
                    self.block_warm_start_draft_hf_config = load_config(
                        self.block_warm_start_draft,
                        trust_remote_code=False,
                    )
                    self.max_model_len = min(
                        self.max_model_len,
                        self.block_warm_start_draft_hf_config.max_position_embeddings,
                    )
            if self.draft_async:
                if self.fan_out_list is None: 
                    self.fan_out_list = [self.async_fan_out] * (self.speculate_k + 1)
                    self.MQ_LEN = sum(self.fan_out_list)
                if self.fan_out_list_miss is None:
                    self.fan_out_list_miss = self.fan_out_list 
                assert sum(self.fan_out_list_miss) == sum(self.fan_out_list), "ERROR in Config: fan_out_list_miss must be the same as fan_out_list"
                
        if self.use_eagle:
            if self.eagle_layers is None:
                L = self.hf_config.num_hidden_layers
                # self.eagle_layers = [3, L//2, L-3]
                self.eagle_layers = [2, L//2, L-3] # [2, 16, 29] outputs, ie. [3, L//2+1, L-2] inputs
                print(f'[Config] just set eagle_layers={self.eagle_layers}', flush=True)
            # Eagle draft must use target's rope_theta (draft config may default to wrong value)
            if self.speculate and self.draft_hf_config is not None:
                target_rope_theta = getattr(self.hf_config, 'rope_theta', 500000.0)
                draft_rope_theta = getattr(self.draft_hf_config, 'rope_theta', 10000.0)
                if target_rope_theta != draft_rope_theta:
                    print(f'[Config] Overriding eagle draft rope_theta: {draft_rope_theta} -> {target_rope_theta}', flush=True)
                    self.draft_hf_config.rope_theta = target_rope_theta
                # Also override max_position_embeddings for correct RoPE cache size
                # NOTE: Do NOT change max_model_len here - it was already correctly capped.
                # Only change draft_hf_config.max_position_embeddings for RoPE.
                target_max_pos = getattr(self.hf_config, 'max_position_embeddings', 8192)
                draft_max_pos = getattr(self.draft_hf_config, 'max_position_embeddings', 2048)
                if target_max_pos != draft_max_pos:
                    print(f'[Config] Overriding eagle draft max_position_embeddings: {draft_max_pos} -> {target_max_pos}', flush=True)
                    self.draft_hf_config.max_position_embeddings = target_max_pos
        
        assert self.max_num_batched_tokens >= self.max_model_len
