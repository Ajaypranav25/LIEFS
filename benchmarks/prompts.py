"""
Fixed benchmark prompt set for LIEFS.

These prompts are used across ALL stages to ensure apples-to-apples
comparisons. We test three scales to expose how performance degrades
with sequence length (especially important for Stage 1 where cost
grows quadratically with output length).

Each prompt is a tuple of (name, user_message, max_new_tokens).
"""

BENCHMARK_PROMPTS: list[tuple[str, str, int]] = [
    (
        "short",
        "What is 2+2?",
        32,
    ),
    (
        "medium",
        "Explain how a transformer model works step by step.",
        128,
    ),
    (
        "long",
        (
            "Write a Python function to implement merge sort with detailed "
            "comments explaining each step of the algorithm."
        ),
        256,
    ),
]
