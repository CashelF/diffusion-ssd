import torch

from ssd.engine.draft_backends import SyncDraftBackendBase
from ssd.engine.sequence import Sequence
from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, SpeculatorBase


class SpeculatorSync(SpeculatorBase):

    def __init__(self, lookahead: int, device: torch.device, draft_backend: SyncDraftBackendBase):
        super().__init__(lookahead, device)
        self.draft_backend = draft_backend

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        print('[spec_prefill] target prefill', flush=True)
        self.draft_backend.prefill(seqs, verify_result)

        if len(seqs) > 0:
            print(
                f"[PREFILL] seq0 prompt_len={seqs[0].num_prompt_tokens} recovery={seqs[0].recovery_token_id}", flush=True)

        return SpeculateResult([], [])

    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        """Generate k speculative tokens using the draft model."""
        assert not verify_result.eagle_acts, "Eagle is not currently supported for synchronous speculation"

        batch_size = len(seqs)
        speculations = torch.zeros(
            batch_size, self.lookahead + 1,
            dtype=torch.int64,
            device=self.device,
        )

        # Single batched write to GPU
        recovery_tokens = []
        for i, seq in enumerate(seqs):
            if seq.recovery_token_id is None:
                raise ValueError(f"recovery_token_id is None for seq {i}")
            recovery_tokens.append(seq.recovery_token_id)
            seq.append_token(seq.recovery_token_id)
        speculations[:, 0] = torch.tensor(
            recovery_tokens, dtype=torch.int64, device=self.device)

        draft_tokens, logits_q = self.draft_backend.draft(seqs, self.lookahead)
        speculations[:, 1:] = draft_tokens

        return SpeculateResult(speculations, logits_q)
