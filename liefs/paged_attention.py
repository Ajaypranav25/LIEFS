"""
Stage 4: Simplified paged attention — block-based KV-cache storage.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class KVBlock:
    """A single block of KV-cache storage.

    Stores K and V tensors for up to `block_size` token positions
    across all layers.
    """
    block_id: int
    block_size: int
    key_cache: torch.Tensor    # (num_layers, num_kv_heads, block_size, head_dim)
    value_cache: torch.Tensor  # (num_layers, num_kv_heads, block_size, head_dim)
    num_filled: int = 0        # How many positions are filled for this block

    @property
    def is_full(self) -> bool:
        return self.num_filled >= self.block_size

    @property
    def free_slots(self) -> int:
        return self.block_size - self.num_filled


class BlockAllocator:
    """Manages a pool of fixed-size KV-cache blocks on GPU."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        if device == "auto" or device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.dtype = dtype

        self.blocks: list[KVBlock] = []
        self.free_block_ids: list[int] = list(range(num_blocks))

        for i in range(num_blocks):
            block = KVBlock(
                block_id=i,
                block_size=block_size,
                key_cache=torch.zeros(
                    num_layers, num_kv_heads, block_size, head_dim,
                    device=self.device, dtype=dtype,
                ),
                value_cache=torch.zeros(
                    num_layers, num_kv_heads, block_size, head_dim,
                    device=self.device, dtype=dtype,
                ),
            )
            self.blocks.append(block)

        bytes_per_block = 2 * num_layers * num_kv_heads * block_size * head_dim * 2
        self.total_memory_mb = num_blocks * bytes_per_block / (1024 * 1024)

    def allocate(self) -> KVBlock | None:
        """Allocate a single free block. Returns None if no blocks available."""
        if not self.free_block_ids:
            return None
        block_id = self.free_block_ids.pop()
        block = self.blocks[block_id]
        block.num_filled = 0
        block.key_cache.zero_()
        block.value_cache.zero_()
        return block

    def free(self, block: KVBlock):
        """Return a block to the free pool."""
        self.free_block_ids.append(block.block_id)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    @property
    def utilization(self) -> float:
        """Memory utilization as a fraction (0.0 to 1.0)."""
        return self.num_used_blocks / self.num_blocks if self.num_blocks > 0 else 0.0


class PagedKVCache:
    """Paged KV-cache for a single sequence."""

    def __init__(self, allocator: BlockAllocator):
        self.allocator = allocator
        self.block_size = allocator.block_size
        self.block_table: list[KVBlock] = []  # Ordered list of blocks
        self.tokens_per_layer = [0] * allocator.num_layers

    @property
    def total_tokens(self) -> int:
        return min(self.tokens_per_layer) if self.tokens_per_layer else 0

    def append_kv(
        self,
        layer_idx: int,
        key: torch.Tensor,    # (1, num_kv_heads, new_tokens, head_dim)
        value: torch.Tensor,  # (1, num_kv_heads, new_tokens, head_dim)
    ):
        """Append K/V tensors for new tokens into the paged blocks."""
        new_tokens = key.shape[2]
        current_layer_tokens = self.tokens_per_layer[layer_idx]

        for t in range(new_tokens):
            tok_idx = current_layer_tokens + t
            block_idx = tok_idx // self.block_size
            pos_in_block = tok_idx % self.block_size

            while block_idx >= len(self.block_table):
                new_block = self.allocator.allocate()
                if new_block is None:
                    raise RuntimeError("Out of KV-cache blocks in BlockAllocator!")
                self.block_table.append(new_block)

            block = self.block_table[block_idx]
            block.key_cache[layer_idx, :, pos_in_block, :] = key[0, :, t, :]
            block.value_cache[layer_idx, :, pos_in_block, :] = value[0, :, t, :]

        self.tokens_per_layer[layer_idx] += new_tokens

        min_toks = min(self.tokens_per_layer)
        for i, b in enumerate(self.block_table):
            filled_in_block = max(0, min(self.block_size, min_toks - i * self.block_size))
            b.num_filled = filled_in_block

    def get_kv(self, layer_idx: int) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Retrieve all cached K/V for a given layer by gathering from blocks."""
        total = self.total_tokens
        if total == 0 or not self.block_table:
            return None, None

        keys = []
        values = []

        remaining = total
        for block in self.block_table:
            if remaining <= 0:
                break
            take = min(self.block_size, remaining)
            keys.append(block.key_cache[layer_idx, :, :take, :])
            values.append(block.value_cache[layer_idx, :, :take, :])
            remaining -= take

        if not keys:
            return None, None

        all_keys = torch.cat(keys, dim=1).unsqueeze(0)    # (1, num_kv_heads, total, head_dim)
        all_values = torch.cat(values, dim=1).unsqueeze(0)

        return all_keys, all_values

    def free_all(self):
        """Free all blocks back to the allocator."""
        for block in self.block_table:
            self.allocator.free(block)
        self.block_table.clear()
        self.tokens_per_layer = [0] * self.allocator.num_layers

    @property
    def num_blocks_used(self) -> int:
        return len(self.block_table)
