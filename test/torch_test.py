import torch
import os
import moe_test
import util
import bench


def count_bytes(*tensors):
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total


def enumerate_m_grouped_contiguous(dtype: torch.dtype):
    for num_groups, num_tokens, hidden_size, inter_size, topk in (
        (1, 1, 16, 16, 1),
        (2, 1, 256, 32, 1),
        (64, 512 * 32, 2880, 2880, 3),
        (4, 35456, 7168, 4096, 1),
    ):
        yield num_groups, num_tokens, hidden_size, inter_size, topk, dtype


def print_error_metric(a, b, type: str):
    is_print = os.getenv("IS_PRINT", False)
    if not is_print:
        return
    print(f"=== Error Metric for {type} ===")
    diff = a - b
    mse = torch.mean(diff**2).item()
    mae = torch.mean(torch.abs(diff)).item()
    rel_err = torch.mean(torch.abs(diff) / (torch.abs(b) + 1e-12)).item() * 100
    print(f"MSE         : {mse:.6e}")
    print(f"MAE         : {mae:.6e}")
    print(f"RelErr (%)  : {rel_err:.4f}%")
    print("d", a)
    print("ref_d_", b)
    print()


def test_dynamic_quantization_moe(
    quant_type: str, quant_value: int, generate_fn, bench_fn, kernel_name: str,
    fused: bool = True
):
    workspace = torch.empty(
        8 * 1024 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )  # 需要保证 workspace 是充足的
    for (
        num_groups,
        num_tokens,
        hidden_size,
        inter_size,
        topk,
        dtype,
    ) in enumerate_m_grouped_contiguous(torch.bfloat16):
        a = torch.normal(
            0, 0.05, size=(num_tokens, hidden_size), device="cuda", dtype=dtype
        )
        b = torch.normal(
            0,
            0.05,
            size=(num_groups, inter_size * 2, hidden_size),
            device="cuda",
            dtype=dtype,
        )
        b1 = torch.normal(
            0,
            0.05,
            size=(num_groups, hidden_size, inter_size),
            device="cuda",
            dtype=dtype,
        )

        scores = torch.randn((num_tokens, num_groups), device="cuda", dtype=dtype)
        _, m_indices = torch.topk(scores, k=topk, dim=1)
        m_indices = m_indices.int()

        a_, b_, b1_, m_indices, d_, ref_d_ = generate_fn(
            num_groups=num_groups,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            inter_size=inter_size,
            topk=topk,
            a=a,
            b=b,
            b1=b1,
            m_indices=m_indices,
        )

        def fn():
            bench_fn(a_, b_, b1_, d_, m_indices, workspace, quant_value, fused)

        t = bench.bench_kineto(
            fn, kernel_name, suppress_kineto_output=True, with_multiple_kernels=True
        )
        if t == 0:
            print(
                f"Eror in perf measurement, set env DG_NSYS_PROFILING=1 to enable nsys profiling"
            )
            continue
        print(
            f"> Grouped Gemm metric for ({num_groups=},num_tokens={num_tokens:5}, hidden_size={hidden_size:6}, "
            f"inter_size={inter_size:5}, {quant_type}, layout=kmajor, fused={fused})".center(100, "=")
        )
        print(
            f"{t * 1e6:4.6f} us | "
            f"{2 * num_tokens * topk * (inter_size * (3 * hidden_size)) / (2 * t) / 1e12:4.0f} TFLOPS | "  # 包含两个 grouped gemm, torch profile 统计的是 相同 kernel name，但不同规模的 kernel 的平均时间
        )
        print_error_metric(d_, ref_d_, quant_type)
        print(f"Moe end to end bench. ")
        print(bench.bench(fn, num_tests=100))
        print()


def compare_grouped_gemm_fused_and_unfused():
    workspace = torch.empty(
        8 * 1024 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )  # 需要保证 workspace 是充足的
    for (
        num_groups,
        num_tokens,
        hidden_size,
        inter_size,
        topk,
        dtype,
    ) in enumerate_m_grouped_contiguous(torch.bfloat16):
        a = torch.normal(
            0, 0.05, size=(num_tokens, hidden_size), device="cuda", dtype=dtype
        )
        b = torch.normal(
            0,
            0.05,
            size=(num_groups, inter_size * 2, hidden_size),
            device="cuda",
            dtype=dtype,
        )
        b1 = torch.normal(
            0,
            0.05,
            size=(num_groups, hidden_size, inter_size),
            device="cuda",
            dtype=dtype,
        )

        scores = torch.randn((num_tokens, num_groups), device="cuda", dtype=dtype)
        _, m_indices = torch.topk(scores, k=topk, dim=1)
        m_indices = m_indices.int()

        a_, b_, b1_, m_indices, d_, ref_d_ = util.generate_channel(
            num_groups=num_groups,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            inter_size=inter_size,
            topk=topk,
            a=a,
            b=b,
            b1=b1,
            m_indices=m_indices,
        )

        moe_test.m_grouped_moe_nt_contiguous_fp8_bf16(a_, b_, b1_, d_, m_indices, workspace, 512, False)
        torch.cuda.synchronize()

        moe_test.m_grouped_moe_nt_contiguous_fp8_bf16(a_, b_, b1_, ref_d_, m_indices, workspace, 512, True)
        torch.cuda.synchronize()

        print_error_metric(d_, ref_d_, "channel")


def test(compute_capability: int):
    test_dynamic_quantization_moe(
        "bfloat16",
        0,
        util.generate_non_quant,
        moe_test.m_grouped_moe_nt_contiguous_bf16,
        "GemmGrouped",
    )
    if compute_capability == 89:
        test_dynamic_quantization_moe(
            "channel",
            512,
            util.generate_channel,
            moe_test.m_grouped_moe_nt_contiguous_fp8_bf16,
            "GemmGrouped",
            fused=False
        )
        test_dynamic_quantization_moe(
            "channel",
            512,
            util.generate_channel,
            moe_test.m_grouped_moe_nt_contiguous_fp8_bf16,
            "GemmGrouped",
            fused=True
        )
    if compute_capability == 90:
        test_dynamic_quantization_moe(
            "block",
            1024,
            util.generate_block,
            moe_test.m_grouped_moe_nt_contiguous_fp8_bf16,
            "GemmUniversal",
        )


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"
    major, minor = torch.cuda.get_device_capability()
    compute_capability = major * 10 + minor

    if compute_capability == 89:
        compare_grouped_gemm_fused_and_unfused()

    test(compute_capability)
