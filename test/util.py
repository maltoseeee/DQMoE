import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

device = "cuda"
block = 128

def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


# 1D1D
def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_amax = x.abs().float().amax(dim=1).view(m, -1).clamp(1e-4)
    sf = x_amax / 448.0
    return (x * (1.0 / sf)).to(torch.float8_e4m3fn).contiguous(), sf


def per_channel_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    e, m, n = x.shape
    x_amax = x.abs().float().amax(dim=2).view(e, m, -1).clamp(1e-4)
    sf = x_amax / 448.0
    return (x * (1.0 / sf)).to(torch.float8_e4m3fn).contiguous(), sf


# 1D2D
def per_token_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    padded_n = align(n, block)
    x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, block)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = x_amax / 448.0
    return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, padded_n)[
        :, :n
    ].contiguous(), sf


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = x.dim()
    x = x[None] if dim == 2 else x
    e, m, n = x.shape
    x_padded = torch.zeros(
        (e, align(m, block), align(n, block)), dtype=x.dtype, device=x.device
    )
    # print(f"x: {x}")
    x_padded[:, :m, :n] = x
    x_view = x_padded.view(
        e, x_padded.size(1) // block, block, x_padded.size(2) // block, block
    )
    x_amax = x_view.abs().float().amax(dim=(2, 4), keepdim=True).clamp(1e-4)
    # print(f"x_amax { x_amax }")
    sf = x_amax / 448.0
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)

    return (
        x_scaled.view_as(x_padded)[:, :m, :n].contiguous(),
        sf.view(e, x_view.size(1), x_view.size(3)),
    )


def torch_moe_grouped(
    a: torch.Tensor,
    b: torch.Tensor,
    b1: torch.Tensor,
    ref_d: torch.Tensor,
    m_indices: torch.Tensor,
    num_groups: int,
):
    for group_idx in range(num_groups):
        idx, top = torch.where(m_indices == group_idx)
        selected_tokens = a[idx]  # [num_selected, hidden_size]

        inter = F.linear(selected_tokens, b[group_idx])  # [num_selected, inter_size*2]

        gate, value = inter.chunk(2, dim=-1)
        activated = F.silu(gate) * value  # [num_selected, inter_size]

        expert_output = F.linear(
            activated, b1[group_idx]
        )  # [num_selected, hidden_size]

        ref_d[idx, top] = expert_output


def fp8_blockwise_torch_moe_grouped(
    a: Tuple[torch.Tensor, torch.Tensor],
    b: Tuple[torch.Tensor, torch.Tensor],
    b1: Tuple[torch.Tensor, torch.Tensor],
    ref_d: torch.Tensor,
    m_indices: torch.Tensor,
    num_groups: int,
):
    """
        模拟 fp8 blockwise moe 的计算
    """
    for group_idx in range(num_groups):
        idx, top = torch.where(m_indices == group_idx)
        if len(idx) == 0:
            continue

        fp8_a = a[0].to(torch.float32)[idx]
        scale_a = a[1][idx]
        a_reduce_size = a[0].shape[-1]

        fp8_b = b[0].to(torch.float32)[group_idx]
        scale_b = b[1][group_idx]
        out_dim_b, in_dim_b = fp8_b.shape


        fp8_b1 = b1[0].to(torch.float32)[group_idx]
        scale_b1 = b1[1][group_idx]
        out_dim_b1, in_dim_b1 = fp8_b1.shape

        scale_a_expanded = scale_a.repeat_interleave(block, dim=-1)[
            :, :a_reduce_size
        ]

        scale_b_expanded = scale_b.repeat_interleave(
            block, dim=-1
        ).repeat_interleave(block, dim=-2)[:out_dim_b, :in_dim_b]
        scale_b1_expanded = scale_b1.repeat_interleave(
            block, dim=-1
        ).repeat_interleave(block, dim=-2)[:out_dim_b1, :in_dim_b1]

        dq_a = fp8_a * scale_a_expanded
        dq_b = fp8_b * scale_b_expanded
        dq_b1 = fp8_b1 * scale_b1_expanded

        inter = F.linear(dq_a, dq_b).to(torch.bfloat16)  # [num_selected, inter_size*2]

        gate, value = inter.chunk(2, dim=-1)
        activated = F.silu(gate) * value  # [num_selected, inter_size]

        [fp8_activated, scale_activated] = per_token_block_cast_to_fp8(activated)
        fp8_activated = fp8_activated.to(torch.float32)
        act_reduce_size = fp8_activated.shape[-1]
        scale_activated_expanded = scale_activated.repeat_interleave(
            block, dim=-1
        )[:, :act_reduce_size]

        dq_activated = fp8_activated * scale_activated_expanded

        # 第二层线性变换
        expert_output = F.linear(
            dq_activated, dq_b1
        ).to(torch.bfloat16)  # [num_selected, hidden_size]

        ref_d[idx, top] = expert_output



def generate_block(
    num_groups: int,
    num_tokens: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    a,
    b,
    b1,
    m_indices,
):
    d = torch.empty(
        (num_tokens, topk, hidden_size), device=device, dtype=torch.bfloat16
    )
    ref_d = torch.empty(
        (num_tokens, topk, hidden_size), device=device, dtype=torch.bfloat16
    )
    torch_moe_grouped(a, b, b1, ref_d, m_indices, num_groups)

    a_fp8 = per_token_block_cast_to_fp8(a)
    b_fp8 = per_block_cast_to_fp8(b)
    b1_fp8 = per_block_cast_to_fp8(b1)

    return a_fp8, b_fp8, b1_fp8, m_indices, d, ref_d


def generate_channel(
    num_groups: int,
    num_tokens: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    a,
    b,
    b1,
    m_indices,
    dtype = torch.bfloat16
):
    d = torch.empty(
        (num_tokens, topk, hidden_size), device=device, dtype=dtype
    )
    ref_d = torch.empty(
        (num_tokens, topk, hidden_size), device=device, dtype=dtype
    )
    torch_moe_grouped(a, b, b1, ref_d, m_indices, num_groups)
    a_fp8 = per_token_cast_to_fp8(a)
    b_fp8 = per_channel_cast_to_fp8(b)
    b1_fp8 = per_channel_cast_to_fp8(b1)

    return a_fp8, b_fp8, b1_fp8, m_indices, d, ref_d

def generate_non_quant(
    num_groups: int,
    num_tokens: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    a,
    b,
    b1,
    m_indices,
    dtype = torch.bfloat16
):
    d = torch.empty(
        (num_tokens, topk, hidden_size), device=device, dtype=dtype
    )
    ref_d = torch.empty(
        (num_tokens, topk, hidden_size), device=device, dtype=dtype
    )
    torch_moe_grouped(a, b, b1, ref_d, m_indices, num_groups)
    a = (a, torch.ones_like(a))
    b = (b, torch.ones_like(b))
    b1 = (b1, torch.ones_like(b1))
    return a, b, b1, m_indices, d, ref_d