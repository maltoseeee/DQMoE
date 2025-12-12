#pragma once
#include <cutlass/util/device_memory.h>
#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>

#include "cuda_fp16.h"
#include "moe_kernels.h"
#include "torch_help.h"
#include "util.h"

struct MoeRunnerKey
{
    size_t num_tokens;
    size_t hidden_size;
    size_t inter_size;
    size_t num_groups;
    size_t topk;
    std::string type_signature;

    bool operator==(MoeRunnerKey const& other) const
    {
        return num_tokens == other.num_tokens && hidden_size == other.hidden_size && inter_size == other.inter_size
            && num_groups == other.num_groups && topk == other.topk && type_signature == other.type_signature;
    }
};

struct MoeRunnerKeyHash
{
    std::size_t operator()(MoeRunnerKey const& key) const
    {
        std::size_t h1 = std::hash<size_t>{}(key.num_tokens);
        std::size_t h2 = std::hash<size_t>{}(key.hidden_size);
        std::size_t h3 = std::hash<size_t>{}(key.inter_size);
        std::size_t h4 = std::hash<size_t>{}(key.num_groups);
        std::size_t h5 = std::hash<size_t>{}(key.topk);
        std::size_t h6 = std::hash<std::string>{}(key.type_signature);

        // Combine hashes
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5);
    }
};

template <typename T, typename WeightType, typename OutputType>
class MoeRunnerCache
{
private:
    static std::unordered_map<MoeRunnerKey, std::shared_ptr<MoeFCRunner<T, WeightType, OutputType>>, MoeRunnerKeyHash>
        cache_;
    static constexpr size_t MAX_CACHE_SIZE = 100;

public:
    static std::shared_ptr<MoeFCRunner<T, WeightType, OutputType>> get_or_create(
        size_t num_tokens, size_t hidden_size, size_t inter_size, size_t num_groups, size_t topk)
    {

        MoeRunnerKey key{num_tokens, hidden_size, inter_size, num_groups, topk,
            std::string(typeid(T).name()) + "_" + std::string(typeid(WeightType).name()) + "_"
                + std::string(typeid(OutputType).name())};

        auto it = cache_.find(key);
        if (it != cache_.end())
        {
            return it->second;
        }

        auto runner = std::make_shared<MoeFCRunner<T, WeightType, OutputType>>(
            num_tokens, hidden_size, inter_size, num_groups, topk);

        if (cache_.size() >= MAX_CACHE_SIZE)
        {
            cache_.erase(cache_.begin());
        }
        cache_[key] = runner;

        return runner;
    }

    static void clear_cache()
    {
        cache_.clear();
    }

    static size_t cache_size()
    {
        return cache_.size();
    }
};

template <typename T, typename WeightType, typename OutputType>
std::unordered_map<MoeRunnerKey, std::shared_ptr<MoeFCRunner<T, WeightType, OutputType>>, MoeRunnerKeyHash>
    MoeRunnerCache<T, WeightType, OutputType>::cache_;

template <typename T,           /*The type used for activations*/
    typename WeightType,        /* The type for the MoE weights */
    typename OutputType = T,    /* The type for the MoE final output */
    typename ScaleType = float, /* The type for scales */
    typename Enable = void>
static void m_grouped_moe_nt_contiguous(std::pair<torch::Tensor, torch::Tensor> const& a,
    std::pair<torch::Tensor, torch::Tensor> const& b, std::pair<torch::Tensor, torch::Tensor> const& b1,
    torch::Tensor d, torch::Tensor const& m_indices, torch::Tensor workspace, int quant_mode_value, bool fused)
{
    auto const& [num_tokens, hidden_size] = get_shape<2>(a.first);
    auto const& [num_groups, inter_size_2, hidden_size_] = get_shape<3>(b.first);
    auto const& [num_tokens_, topk, hidden_size__] = get_shape<3>(d);

    auto const& [num_groups_, hidden_size___, inter_size] = get_shape<3>(b1.first);
    auto const& [num_tokens___, topk_] = get_shape<2>(m_indices);

    CPU_CHECK(num_tokens == num_tokens_ and num_groups == num_groups_ and hidden_size == hidden_size_
        and hidden_size == hidden_size__ and hidden_size == hidden_size___ and inter_size * 2 == inter_size_2);

    // CPU_CHECK(n > 0 and k > 0 and num_groups > 0);
    // CPU_CHECK(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    // CPU_CHECK(b.first.scalar_type() == torch::kFloat8_e4m3fn);
    // CPU_CHECK(d.scalar_type() == torch::kBFloat16 or d.scalar_type() ==
    // torch::kFloat or d.scalar_type() == torch::kHalf);
    CPU_CHECK(m_indices.scalar_type() == torch::kInt);

    if (num_tokens == 0)
        return;

    auto quant_mode = static_cast<QuantMode>(quant_mode_value);

    auto const& sfa = a.second;
    auto const& sfb = b.second;
    auto const& sfb1 = b1.second;

    auto moe_runner = MoeRunnerCache<T, WeightType, OutputType>::get_or_create(
        num_tokens, hidden_size, inter_size, num_groups, topk);

    size_t workspace_size
        = moe_runner->getWorkspaceSize(num_tokens, hidden_size, inter_size, num_groups, topk, quant_mode);

    // cutlass::DeviceAllocation<char> workspace(workspace_size);
    CPU_CHECK_FORMAT(
        is_aligned(workspace.data_ptr(), CUDA_MEM_ALIGN), "workspace pointer %p is not aligned!", workspace.data_ptr());
    // cudaStream_t stream;
    // cudaStreamCreate(&stream);
    auto stream = at::cuda::getCurrentCUDAStream();
    moe_runner->runMoeWithTorchDebug(a.first.data_ptr(), b.first.data_ptr(), b1.first.data_ptr(),
        static_cast<int*>(m_indices.data_ptr()), nullptr, nullptr, d.data_ptr(), quant_mode,
        QuantParams::FP8(static_cast<float*>(sfa.data_ptr()), static_cast<float*>(sfb.data_ptr()),
            static_cast<float*>(sfb1.data_ptr())),
        num_groups, topk, num_tokens, hidden_size, inter_size, static_cast<char*>(workspace.data_ptr()), stream, {}, fused);
}

static void register_apis(pybind11::module_& m)
{
    m.def("get_mn_major_tma_aligned_tensor", &get_mn_major_tma_aligned_tensor);
    m.def("m_grouped_moe_nt_contiguous_fp8_bf16",
        &m_grouped_moe_nt_contiguous<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>, py::arg("a"), py::arg("b"),
        py::arg("b1"), py::arg("d"), py::arg("m_indices"), py::arg("workspace"), py::arg("quant_mode_value"),
        py::arg("fused") = true);
    m.def("m_grouped_moe_nt_contiguous_bf16", &m_grouped_moe_nt_contiguous<__nv_bfloat16, __nv_bfloat16>, py::arg("a"),
        py::arg("b"), py::arg("b1"), py::arg("d"), py::arg("m_indices"), py::arg("workspace"),
        py::arg("quant_mode_value"), py::arg("fused") = true);
}