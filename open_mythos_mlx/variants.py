from .model import MythosConfig


def mythos_1b() -> MythosConfig:
    """1B parameter config. dim=2048, 64 experts, 16 loop iters, 4k context."""
    return MythosConfig(
        vocab_size=32000, dim=2048, n_heads=16, n_kv_heads=4,
        max_seq_len=4096, max_loop_iters=16, prelude_layers=2, coda_layers=2,
        attn_type="mla", kv_lora_rank=256, q_lora_rank=512,
        qk_rope_head_dim=32, qk_nope_head_dim=64, v_head_dim=64,
        n_experts=64, n_shared_experts=2, n_experts_per_tok=4, expert_dim=2048,
        lora_rank=8,
    )


def mythos_3b() -> MythosConfig:
    """3B parameter config. dim=3072, 64 experts, 16 loop iters, 4k context."""
    return MythosConfig(
        vocab_size=32000, dim=3072, n_heads=24, n_kv_heads=6,
        max_seq_len=4096, max_loop_iters=16, prelude_layers=2, coda_layers=2,
        attn_type="mla", kv_lora_rank=384, q_lora_rank=768,
        qk_rope_head_dim=32, qk_nope_head_dim=96, v_head_dim=96,
        n_experts=64, n_shared_experts=2, n_experts_per_tok=4, expert_dim=4096,
        lora_rank=8,
    )


def mythos_10b() -> MythosConfig:
    """10B parameter config. dim=4096, 128 experts, 24 loop iters, 8k context."""
    return MythosConfig(
        vocab_size=32000, dim=4096, n_heads=32, n_kv_heads=8,
        max_seq_len=8192, max_loop_iters=24, prelude_layers=2, coda_layers=2,
        attn_type="mla", kv_lora_rank=512, q_lora_rank=1024,
        qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128,
        n_experts=128, n_shared_experts=2, n_experts_per_tok=4, expert_dim=4096,
        lora_rank=16,
    )
