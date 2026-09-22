"""
Stage 3: Request scheduler for continuous batching.

WHAT IS CONTINUOUS BATCHING?
============================

In STATIC batching, you collect N requests, process them together until ALL
finish, then collect the next N. This wastes GPU time because short sequences
sit idle while long ones finish.

In CONTINUOUS batching (also called "iteration-level batching" or "inflight
batching"), when a sequence finishes, its slot is IMMEDIATELY filled with
a new request from the queue. The GPU stays busy and throughput improves.
"""

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

import torch

from liefs.utils import get_eos_token_ids


class RequestStatus(Enum):
    """Lifecycle states for a generation request."""
    QUEUED = auto()
    PREFILLING = auto()
    GENERATING = auto()
    FINISHED = auto()


@dataclass
class GenerationRequest:
    """A single generation request with its state."""
    request_id: str
    input_ids: torch.Tensor           # (1, prompt_len) on CUDA
    max_new_tokens: int
    status: RequestStatus = RequestStatus.QUEUED

    # Populated during generation
    generated_ids: list[int] = field(default_factory=list)
    kv_cache: object = None           # DynamicCache or tuple of (K, V)
    current_logits: torch.Tensor | None = None  # (1, vocab_size)
    finish_reason: str = ""           # "stop" (EOS) or "length" (max tokens)

    # Timing
    created_at: float = 0.0
    first_token_at: float = 0.0
    finished_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.perf_counter()

    @property
    def tokens_generated(self) -> int:
        return len(self.generated_ids)


class ContinuousBatchScheduler:
    """Scheduler that manages concurrent generation requests."""

    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size: int = 4,
        eos_token_ids: set[int] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size

        # EOS tokens
        self.eos_token_ids: set[int] = (
            eos_token_ids if eos_token_ids is not None else get_eos_token_ids(tokenizer)
        )

        # Request management
        self.queue: deque[GenerationRequest] = deque()
        self.active_slots: list[GenerationRequest] = []
        self.completed: list[GenerationRequest] = []

    def add_request(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        request_id: str | None = None,
    ) -> str:
        """Add a new generation request to the queue."""
        if request_id is None:
            request_id = str(uuid.uuid4())[:8]

        request = GenerationRequest(
            request_id=request_id,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )
        self.queue.append(request)
        return request_id

    def _fill_slots(self):
        """Move queued requests into active slots if space is available."""
        while len(self.active_slots) < self.max_batch_size and self.queue:
            request = self.queue.popleft()
            request.status = RequestStatus.PREFILLING
            self.active_slots.append(request)

    @torch.no_grad()
    def _prefill_request(self, request: GenerationRequest):
        """Run the prefill phase for a single request."""
        outputs = self.model(
            input_ids=request.input_ids,
            use_cache=True,
        )
        request.kv_cache = outputs.past_key_values
        request.current_logits = outputs.logits[:, -1, :]
        request.status = RequestStatus.GENERATING

    @torch.no_grad()
    def _decode_step(self, request: GenerationRequest):
        """Run one decode step for a single request."""
        next_token_id = request.current_logits.argmax(dim=-1).item()

        if next_token_id in self.eos_token_ids:
            request.status = RequestStatus.FINISHED
            request.finish_reason = "stop"
            request.finished_at = time.perf_counter()
            return

        request.generated_ids.append(next_token_id)

        if len(request.generated_ids) == 1:
            request.first_token_at = time.perf_counter()

        if request.tokens_generated >= request.max_new_tokens:
            request.status = RequestStatus.FINISHED
            request.finish_reason = "length"
            request.finished_at = time.perf_counter()
            return

        next_token_tensor = torch.tensor(
            [[next_token_id]],
            device=request.input_ids.device,
            dtype=request.input_ids.dtype,
        )
        outputs = self.model(
            input_ids=next_token_tensor,
            past_key_values=request.kv_cache,
            use_cache=True,
        )
        request.kv_cache = outputs.past_key_values
        request.current_logits = outputs.logits[:, -1, :]

    def _remove_finished(self):
        """Move finished requests from active slots to completed list,
        freeing their KV-cache memory."""
        still_active = []
        for req in self.active_slots:
            if req.status == RequestStatus.FINISHED:
                req.kv_cache = None  # Free GPU memory
                self.completed.append(req)
            else:
                still_active.append(req)
        self.active_slots = still_active

    def run(self) -> list[GenerationRequest]:
        """Run all queued requests with continuous batching."""
        self.completed = []

        while self.queue or self.active_slots:
            self._fill_slots()

            if not self.active_slots:
                break

            for req in self.active_slots:
                if req.status == RequestStatus.PREFILLING:
                    self._prefill_request(req)
                elif req.status == RequestStatus.GENERATING:
                    self._decode_step(req)

            self._remove_finished()

        return self.completed

    def reset(self):
        """Clear all state for a fresh run."""
        self.queue.clear()
        self.active_slots.clear()
        self.completed.clear()
