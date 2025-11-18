#pragma once

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/python.h>
#include <torch/torch.h>

#include <tuple>

// Tensor utils
template <int N>
static auto get_shape(torch::Tensor const& t)
{
    return [&t]<size_t... Is>(std::index_sequence<Is...>)
    { return std::make_tuple(static_cast<int>(t.sizes()[Is])...); }(std::make_index_sequence<N>());
}

static int get_tma_aligned_size(int const& x, int const& element_size)
{
    return x;
}

static std::tuple<int, int, int, int, int, torch::Tensor> preprocess_sf(torch::Tensor const& sf)
{
    // NOTES: for the extreme performance, you may rewrite/fuse this function in
    // CUDA
    auto const& dim = sf.dim();
    CHECK(dim == 2 or dim == 3);
    CHECK(sf.scalar_type() == torch::kFloat);
    auto const& batched_sf = dim == 2 ? sf.unsqueeze(0) : sf;

    auto const& [num_groups, mn, sf_k] = get_shape<3>(batched_sf);
    auto const& tma_aligned_mn = get_tma_aligned_size(mn, static_cast<int>(sf.element_size()));
    return {dim, num_groups, mn, sf_k, tma_aligned_mn, batched_sf};
}

static torch::Tensor get_mn_major_tma_aligned_tensor(torch::Tensor const& sf)
{
    auto const& [dim, num_groups, mn, sf_k, tma_aligned_mn, batched_sf] = preprocess_sf(sf);

    // The last kernel already gives a column-major TMA aligned layout
    if ((batched_sf.stride(0) == tma_aligned_mn * sf_k or dim == 2) and batched_sf.stride(1) == 1
        and batched_sf.stride(2) == tma_aligned_mn)
        return (dim == 2) ? batched_sf.squeeze(0) : batched_sf;

    auto const& out = torch::empty_strided(
        {num_groups, mn, sf_k}, {tma_aligned_mn * sf_k, 1, tma_aligned_mn}, batched_sf.options());

    out.copy_(batched_sf);

    return (dim == 2) ? out.squeeze(0) : out;
}

int align(int n, int block_size)
{
    return (n + block_size - 1) / block_size * block_size;
}

std::tuple<torch::Tensor, torch::Tensor> per_token_block_cast_to_fp8(torch::Tensor x)
{
    assert(x.dim() == 2);

    int m = x.size(0);
    int n = x.size(1);
    int padded_n = align(n, 128);

    auto options = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto x_padded = torch::zeros({m, padded_n}, options);

    x_padded.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, n)}, x);
    auto x_view = x_padded.view({m, -1, 128});

    auto x_amax = x_view.abs().to(torch::kFloat).amax(2).view({m, -1}).clamp(1e-4);

    auto sf = x_amax / 448.0;

    auto scaled = x_view * (1.0 / sf.unsqueeze(2));
    auto fp8_output = scaled.to(torch::kFloat8_e4m3fn)
                          .view({m, padded_n})
                          .index({torch::indexing::Slice(), torch::indexing::Slice(0, n)})
                          .contiguous();

    return std::make_tuple(fp8_output, sf);
}