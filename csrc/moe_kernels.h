#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped_per_group_scale.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cub/cub.cuh>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/tensor_ref.h"
#include "moe_gemm_template.h"
#include "quantization.h"
#include "torch_help.h"
#include "util.h"

static size_t cubHistogramAndScan(void* workspace, size_t workspace_size, int const* d_indices, int* d_counts,
    int* d_cumsum, int num_elements, int num_experts, cudaStream_t stream = 0);

// Main FP8 MOE Runner class
template <typename T,           /*The type used for activations*/
    typename WeightType,        /* The type for the MoE weights */
    typename OutputType = T,    /* The type for the MoE final output */
    typename ScaleType = float, /* The type for scales */
    typename Enable = void>
class MoeFCRunner
{
public:
    static constexpr bool use_fp8 = (std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>);
    MoeFCRunner(int num_tokens, int hidden_size, int inter_size, int num_experts, int k);
    ~MoeFCRunner() = default;

    size_t getWorkspaceSize(
        int num_tokens, int hidden_size, int inter_size, int num_experts, int k, QuantMode quant_mode) const;
    void runMoe(void const* expert_inputs, void const* expert_weights1, void const* expert_weights2,
        int const* topk_indices, void const* expert_bias1, void const* expert_bias2, void* final_output,
        QuantMode quant_mode, QuantParams quant_param, int num_experts, int topk, int num_tokens, int hidden_size,
        int inter_size, char* workspace, cudaStream_t stream);

    void runMoeWithTorchDebug(void const* expert_inputs, void const* expert_weights1, void const* expert_weights2,
        int const* topk_indices, void const* expert_bias1, void const* expert_bias2, void* final_output,
        QuantMode quant_mode, QuantParams quant_params, int num_experts, int topk, int num_tokens, int hidden_size,
        int inter_size, char* workspace, cudaStream_t stream,
        std::map<std::string, std::pair<void*, size_t>> intermediate_map);

private:
    size_t gemm_workspace_size_;
    int per_token_grid_size_ = 128; // for launch cooperative kernel FIXME(zengliang03)

    MoeGemmRunner<T, WeightType, OutputType, ScaleType> moe_gemm_runner_;
    void Gemm(T const* expert_inputs, WeightType const* expert_weights, int const* d_counts,
        int const* d_cumsum_counter, OutputType* gemm_output, float* pending_dq_output, QuantMode const& quant_mode,
        ScaleType const* scale_a, ScaleType const* scale_b, int num_experts, int num_tokens, int topk, int N, int K,
        char* workspace, size_t workspace_size, std::string operation_name, cudaStream_t stream);

    void extractIntermediateResultsforTorch(char* workspace, int num_experts, int topk, int bs, int hidden_size,
        int inter_size, QuantMode quant_mode, std::map<std::string, std::pair<void*, size_t>> intermediate_map,
        cudaStream_t stream);

    std::vector<cutlass::gemm::GemmCoord> problem_sizes_host_;

private:
    std::map<std::string, std::pair<size_t, size_t>> getWorkspaceDeviceBufferSizes(
        int num_tokens, int hidden_size, int inter_size, int num_experts, int k, QuantMode quant_mode) const;
};

// ============================================================================
// KERNEL IMPLEMENTATIONS
// ============================================================================

size_t cubHistogramAndScan(void* workspace, size_t workspace_size, int const* d_indices, int* d_counts, int* d_cumsum,
    int num_elements, int num_experts, cudaStream_t stream)
{
    if (workspace == nullptr)
    {
        size_t temp_storage_bytes = 0, temp_storage_bytes2 = 0;
        cub::DeviceHistogram::HistogramEven(
            nullptr, temp_storage_bytes, d_indices, d_counts, num_experts + 1, 0, num_experts, num_elements, stream);
        cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes2, d_counts, d_cumsum, num_experts + 1, stream);

        return max(temp_storage_bytes, temp_storage_bytes2);
    }
    // Compute histogram using CUB's DeviceHistogram
    cub::DeviceHistogram::HistogramEven(
        workspace, workspace_size, d_indices, d_counts, num_experts + 1, 0, num_experts, num_elements, stream);

    // Compute exclusive scan on counts
    cub::DeviceScan::ExclusiveSum(workspace, workspace_size, d_counts, d_cumsum, num_experts + 1, stream);
    return 0;
}

// ===================================
// Dynamic Quantization
// ===================================

constexpr static int QUANTIZE_THREADS_PER_BLOCK = 256;

// Improved per-tensor scale combination kernel using cooperative groups
template <typename InputType, typename OutputType>
__global__ void perTensorDynamicFp8QuantKernel(InputType const* input_data, // Input tensor data for scale computation
    float* per_tensor_scale_out,                                            // Output per-tensor scale [2], scale[0]
                                 // 实际输出，scale[1] 用于临时的 max_val 辅助存储
    OutputType* output_data, int num_experts, int axis_size, int expanded_token_size)
{
    namespace cg = cooperative_groups;

    // Create cooperative groups
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    auto block_num = gridDim.x;

    int const tid = threadIdx.x;
    int const bid = blockIdx.x;
    int const start_offset = threadIdx.x;

    // Use vectorized loads similar to dynamicScaledFp8QuantKernel
    constexpr int ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<InputType>::value;
    int const stride = QUANTIZE_THREADS_PER_BLOCK;
    // const int num_elems_vec = axis_size / ELEM_PER_THREAD;

    using DataElem = cutlass::Array<InputType, ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, ELEM_PER_THREAD>;
    using OutputElem = cutlass::Array<OutputType, ELEM_PER_THREAD>;

    int const ROW_PER_BLOCK = (expanded_token_size + block_num - 1) / block_num;
    int const real_row_per_block = min(ROW_PER_BLOCK, expanded_token_size - bid * ROW_PER_BLOCK);
    int const num_elems_vec = real_row_per_block * axis_size / ELEM_PER_THREAD;
    auto input_vec = reinterpret_cast<DataElem const*>(input_data + bid * ROW_PER_BLOCK * axis_size);
    auto output_vec = reinterpret_cast<OutputElem*>(output_data + bid * ROW_PER_BLOCK * axis_size);
    // Find local maximum
    float local_max = 0.0f;
    cutlass::maximum_absolute_value_reduction<ComputeElem> amax;

#pragma unroll
    for (int elem_idx = start_offset; elem_idx < num_elems_vec; elem_idx += stride)
    {
        local_max = amax(local_max, arrayConvert<DataElem, ComputeElem>(input_vec[elem_idx]));
    }

    // Block-level reduction using CUB
    using BlockReduce = cub::BlockReduce<float, QUANTIZE_THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_max = BlockReduce(temp_storage).Reduce(local_max, cuda::maximum<>{});
    __syncthreads();
    // Update global maximum using atomic operation
    if (tid == 0)
    {
        atomicMax(reinterpret_cast<int*>(per_tensor_scale_out + 1), __float_as_int(block_max));
    }
    // Grid-level synchronization to ensure per-tensor scale is computed
    grid.sync();

    float per_tensor_inv_scale = (per_tensor_scale_out[1] == 0.F) ? 0.F : (448.F / (per_tensor_scale_out[1]));
#pragma unroll
    for (int elem_index = start_offset; elem_index < num_elems_vec; elem_index += stride)
    {
        auto value = arrayConvert<DataElem, ComputeElem>(input_vec[elem_index]);
        output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(value * per_tensor_inv_scale);
    }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
    {
        per_tensor_scale_out[0] = per_tensor_scale_out[1] / 448.F;
    }
}

template <typename InputType, typename OutputType>
void launchTensorDynamicFp8Quant(InputType const* input_data, float* per_tensor_scale_out, OutputType* output_data,
    int num_experts, int axis_size, int expanded_token_size, int per_token_grid_size, cudaStream_t stream)
{
    constexpr int QUANTIZE_THREADS_PER_BLOCK = 256;
    constexpr int QUANTIZE_BLOCKS_PER_GRID = 512;
    dim3 block(QUANTIZE_THREADS_PER_BLOCK);
    dim3 grid(QUANTIZE_BLOCKS_PER_GRID);

    // Launch cooperative kernel
    void* kernel_args[] = {(void*) &input_data, (void*) &per_tensor_scale_out, (void*) &output_data,
        (void*) &num_experts, (void*) &axis_size, (void*) &expanded_token_size};

    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*) perTensorDynamicFp8QuantKernel<InputType, OutputType>, grid, block, kernel_args, 0, stream));
}

template <typename InputType, typename OutputType>
__global__ void perTokendynamicFp8QuantKernel(
    InputType const* input, OutputType* output, float* scale_out, int const axis_size)
{
    int64_t const tid = threadIdx.x;
    int64_t const token_idx = blockIdx.x;

    constexpr int64_t QUANTIZE_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<InputType>::value;
    int64_t const start_offset = threadIdx.x;
    int64_t const stride = QUANTIZE_THREADS_PER_BLOCK;
    int64_t const num_elems_in_col = axis_size / QUANTIZE_ELEM_PER_THREAD;

    InputType const* row_in = input + token_idx * axis_size;
    OutputType* row_out = output + token_idx * axis_size;

    using DataElem = cutlass::Array<InputType, QUANTIZE_ELEM_PER_THREAD>;
    using OutputElem = cutlass::Array<OutputType, QUANTIZE_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, QUANTIZE_ELEM_PER_THREAD>;
    auto row_in_vec = reinterpret_cast<DataElem const*>(row_in);
    auto row_out_vec = reinterpret_cast<OutputElem*>(row_out);

    float local_max{0.0F};
    cutlass::maximum_absolute_value_reduction<ComputeElem> amax;
#pragma unroll
    for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        local_max = amax(local_max, arrayConvert<DataElem, ComputeElem>(row_in_vec[elem_index]));
    }

    // calculate for absmax
    using BlockReduce = cub::BlockReduce<float, QUANTIZE_THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage tmp;
    float block_max = BlockReduce(tmp).Reduce(local_max, cuda::maximum<>{});
    __shared__ float shared_max;
    if (tid == 0)
    {
        scale_out[blockIdx.x] = static_cast<float>(block_max) / 448.F;
        shared_max = static_cast<float>(block_max);
    }
    __syncthreads();
    float inv_s = (shared_max == 0.F) ? 0.F : (448.F / shared_max);

#pragma unroll
    for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto value = arrayConvert<DataElem, ComputeElem>(row_in_vec[elem_index]);
        row_out_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(value * inv_s);
    }
}

template <typename InputType, typename OutputType>
void launchTokenDynamicFp8Quant(InputType const* input, OutputType* output, float* scale_out, int const num_elements,
    int const axis_size, cudaStream_t stream)
{
    dim3 grid(num_elements);
    dim3 block(QUANTIZE_THREADS_PER_BLOCK);
    perTokendynamicFp8QuantKernel<InputType, OutputType>
        <<<grid, block, 0, stream>>>(input, output, scale_out, axis_size);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__device__ __host__ constexpr T div_up(T a, int b)
{
    return (a + b - 1) / b;
}

template <typename T>
__forceinline__ __device__ T find_max_elem_in_warp(T value)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        value = T(std::max(float(value), __shfl_down_sync(0xFFFFFFFF, float(value), offset)));
    }
    value = T(__shfl_sync(0xffffffff, float(value), 0));
    return value;
}

template <typename InputType, typename OutputType, typename ScaleType = float>
__global__ void scale_1x128_kernel(
    OutputType* output, ScaleType* scales, InputType const* const input, int dim_x, int dim_y)
{
    size_t scales_along_dim_x = div_up(dim_x, 128);
    size_t scales_along_dim_y = div_up(dim_y, 1);
    using Input2Type = typename std::conditional_t<std::is_same_v<InputType, float>, float2,
        std::conditional_t<std::is_same_v<InputType, half>, half2, __nv_bfloat162>>;
    for (size_t warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
        warp_idx < scales_along_dim_x * scales_along_dim_y; warp_idx += gridDim.x * blockDim.x / 32)
    {
        int scales_idx_y = warp_idx / scales_along_dim_x;
        int scales_idx_x = warp_idx % scales_along_dim_x;

        InputType const* input_line = input + (size_t) scales_idx_y * dim_x + scales_idx_x * 128;
        InputType input_amax = InputType(0);
        // Each thread reads 2 elements from input_line
        int lane_id = threadIdx.x % 32 * 2;

        Input2Type input_frag2[2] = {Input2Type(0, 0), Input2Type(0, 0)};
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x)
            {
                break;
            }
            else
            {
                input_frag2[i] = *((Input2Type*) (input_line) + lane_id / 2);
            }
            input_line += 64;
        }
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x)
            {
                break;
            }
            else
            {
                if constexpr (std::is_same_v<InputType, float>)
                {
                    input_amax = InputType(fmax(input_amax, fmax(fabs(input_frag2[i].x), fabs(input_frag2[i].y))));
                }
                else
                {
                    input_amax
                        = InputType(__hmax(input_amax, __hmax(__habs(input_frag2[i].x), __habs(input_frag2[i].y))));
                }
            }
        }
        InputType amax = find_max_elem_in_warp(input_amax);
        ScaleType scale = amax != InputType(0.f) ? 448.f / ScaleType(amax) : 1.f;

        if (lane_id == 0)
        {
            scales[(size_t) scales_idx_y * scales_along_dim_x + scales_idx_x] = ScaleType(1.f / scale);
        }

        OutputType* output_line = output + (size_t) scales_idx_y * dim_x + scales_idx_x * 128;
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x)
            {
                break;
            }
            else
            {
                ScaleType value_1 = ScaleType(input_frag2[i].x) * scale;
                ScaleType value_2 = ScaleType(input_frag2[i].y) * scale;
                output_line[lane_id] = OutputType(value_1);
                output_line[lane_id + 1] = OutputType(value_2);
            }
            output_line += 64;
        }
    }
}

template <typename InputType, typename OutputType>
void launchBlockDynamicFp8Quant(InputType const* input, OutputType* output, float* scale_out, int const num_elements,
    int const axis_size, cudaStream_t stream)
{
    // auto torch_stream = c10::cuda::getStreamFromExternal(stream, 0);
    // at::cuda::CUDAStreamGuard guard(torch_stream);
    //   auto input_type = torch::kFloat32;
    //   if constexpr (std::is_same_v<InputType, float>) {
    //     input_type = torch::kFloat32;
    //   } else if constexpr (std::is_same_v<InputType, half>) {
    //     input_type = torch::kFloat16;
    //   } else if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
    //     input_type = torch::kBFloat16;
    //   }

    scale_1x128_kernel<<<num_elements, 256, 0, stream>>>(output, scale_out, input, axis_size, num_elements);
}

// =================================
//  Gated Activation
// =================================

constexpr static int ACTIVATION_THREADS_PER_BLOCK = 256;

template <class T, template <class> class ActFn>
__global__ void doGatedActivationKernel(T const* gemm_result, T* output, int64_t inter_size)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;

    output = output + token * inter_size;
    gemm_result = gemm_result + token * inter_size * 2;

    constexpr int64_t ACTIVATION_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<T>::value;

    using DataElem = cutlass::Array<T, ACTIVATION_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, ACTIVATION_ELEM_PER_THREAD>;
    auto gemm_result_vec = reinterpret_cast<DataElem const*>(gemm_result);
    auto output_vec = reinterpret_cast<DataElem*>(output);
    int64_t const start_offset = tid;
    int64_t const stride = ACTIVATION_THREADS_PER_BLOCK;
    assert(inter_size % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / ACTIVATION_ELEM_PER_THREAD;
    int64_t const inter_size_vec = inter_size / ACTIVATION_ELEM_PER_THREAD;

    ActFn<ComputeElem> fn{};
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        // BF16 isn't supported, use FP32 for activation function
        auto gate_value = arrayConvert<DataElem, ComputeElem>(gemm_result_vec[elem_index]);
        auto gate_act = fn(gate_value);
        auto fc2_value = arrayConvert<DataElem, ComputeElem>(gemm_result_vec[elem_index + inter_size_vec]);
        output_vec[elem_index] = arrayConvert<ComputeElem, DataElem>(gate_act * fc2_value);
    }
}

template <typename T>
void launchGatedActivation(T const* input, T* output, int num_elements, int inter_size, cudaStream_t stream)
{
    dim3 block(ACTIVATION_THREADS_PER_BLOCK);
    dim3 grid(num_elements);
    doGatedActivationKernel<T, cutlass::epilogue::thread::SiLu><<<grid, block, 0, stream>>>(input, output, inter_size);
}

// topk_indices: [bs, topk_num]
// cumsum_counter: [num_experts]
// expanded_dest_row_to_expanded_source_row: [bs * topk_num], unpermute
// indices map to permute indices per_token_dest_row_start_indices
// 是一个辅助数组，每个 token expand 之后的 indices 在 permute 后的
// indices
__global__ void GetGroupIndex(int const* topk_indices, int const* cumsum_counter,
    int* expanded_dest_row_to_expanded_source_row, int num_elements, int topk, int* per_token_dest_row_start_indices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements)
        return;
    int expert_index = topk_indices[idx];
    int pos = atomicAdd(&per_token_dest_row_start_indices[expert_index], 1);
    expanded_dest_row_to_expanded_source_row[cumsum_counter[expert_index] + pos] = idx;
}

void launchGetGroupedIndex(int const* topk_indices, int const* cumsum_counter,
    int* expanded_dest_row_to_expanded_source_row, int num_elements, int topk, int* per_token_dest_row_start_indices,
    cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    GetGroupIndex<<<grid_size, block_size, 0, stream>>>(topk_indices, cumsum_counter,
        expanded_dest_row_to_expanded_source_row, num_elements, topk, per_token_dest_row_start_indices);
}

// 为 MoE（混合专家）机制复制并重排行数据。

// “expanded_x_row” 表示扩展后的行数为 num_rows × k。
// 之所以称为“扩展”，是因为我们需要对输入矩阵中的某些行进行复制，以匹配目标维度。
// 这些重复的行最终总会被路由到不同的专家（expert）中。

// 注意：此处提到的 expanded_dest_row_to_expanded_source_row
// 映射表中的索引范围为 (0, k * rows_in_input - 1)。
// 该映射表的构造方式确保了： 索引 rows_in_input、rows_in_input +
// 1、...、rows_in_input + k - 1 都对应原始输入矩阵中的第 0 行。
// 因此，要确定在源矩阵中应从哪一行读取数据，只需对扩展后的索引取模（即对
// k 取模）即可。
constexpr static int EXPAND_THREADS_PER_BLOCK = 256;

template <typename T>
__global__ void expandInputRowsGatherKernel(T const* unpermuted_input,
    int const* expanded_dest_row_to_expanded_source_row, T* permuted_output, int64_t const num_rows, int64_t const cols,
    int64_t k)
{
    int64_t const expanded_dest_row = blockIdx.x;
    int64_t const expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];

    // Load 128-bits per thread
    constexpr int64_t ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<T>::value;
    using DataElem = cutlass::Array<T, ELEM_PER_THREAD>;

    // Duplicate and permute rows
    int64_t const source_row = expanded_source_row / k;

    auto const* source_row_ptr = reinterpret_cast<DataElem const*>(unpermuted_input + source_row * cols);
    auto* dest_row_ptr = reinterpret_cast<DataElem*>(permuted_output + expanded_dest_row * cols);

    int64_t const start_offset = threadIdx.x;
    int64_t const stride = EXPAND_THREADS_PER_BLOCK;
    int64_t const num_elems_in_col = cols / ELEM_PER_THREAD;

#pragma unroll
    for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        dest_row_ptr[elem_index] = source_row_ptr[elem_index];
    }
}

template <typename T>
void launchExpandInputRowsGatherKernel(T const* unpermuted_input, int const* expanded_dest_row_to_expanded_source_row,
    T* permuted_output, int64_t const num_rows, int64_t const cols, int64_t k, cudaStream_t stream)
{
    int64_t const blocks = num_rows * k;
    int64_t const threads = EXPAND_THREADS_PER_BLOCK;

    expandInputRowsGatherKernel<<<blocks, threads, 0, stream>>>(
        unpermuted_input, expanded_dest_row_to_expanded_source_row, permuted_output, num_rows, cols, k);
}

__global__ void expandInputScaleGatherKernel(float const* unpermuted_scale,
    int const* expanded_dest_row_to_expanded_source_row, float* permuted_output, int64_t const num_elements, int64_t k)
{
    int64_t const expanded_dest_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (expanded_dest_row >= num_elements)
        return;
    int64_t const expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
    int64_t const source_row = expanded_source_row / k;
    permuted_output[expanded_dest_row] = unpermuted_scale[source_row];
}

void launchExpandInputScaleGatherKernel(float const* unpermuted_scale,
    int const* expanded_dest_row_to_expanded_source_row, float* permuted_output, int64_t const num_rows, int64_t k,
    cudaStream_t stream)
{
    constexpr int block_size = 256;
    int64_t const grid_size = (num_rows * k + block_size - 1) / block_size;
    expandInputScaleGatherKernel<<<grid_size, block_size, 0, stream>>>(
        unpermuted_scale, expanded_dest_row_to_expanded_source_row, permuted_output, num_rows * k, k);
}

__global__ void expandInputScaleBlockGatherKernel(float const* unpermuted_scale,
    int const* expanded_dest_row_to_expanded_source_row, float* permuted_output, int64_t const num_elements, int cols,
    int64_t k)
{
    int64_t const expanded_dest_row = blockIdx.x;
    int64_t const expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
    int64_t const source_row = expanded_source_row / k;
    auto const* source_row_ptr = unpermuted_scale + source_row * cols;
    auto* dest_row_ptr = permuted_output + expanded_dest_row * cols;

    int64_t const start_offset = threadIdx.x;
    int64_t const stride = EXPAND_THREADS_PER_BLOCK;
    int64_t const num_elems_in_col = cols;
#pragma unroll
    for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        dest_row_ptr[elem_index] = source_row_ptr[elem_index];
    }
}

void launchExpandInputScaleBlockGatherKernel(float const* unpermuted_scale,
    int const* expanded_dest_row_to_expanded_source_row, float* permuted_output, int64_t const num_rows,
    int64_t const cols, int64_t k, cudaStream_t stream)
{
    int64_t const blocks = num_rows * k;
    int64_t const threads = EXPAND_THREADS_PER_BLOCK;
    expandInputScaleBlockGatherKernel<<<blocks, threads, 0, stream>>>(
        unpermuted_scale, expanded_dest_row_to_expanded_source_row, permuted_output, num_rows, cols, k);
}

constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;

// Final kernel to unpermute
template <typename T>
__global__ void finalizeScatterKernel(T const* expanded_permuted_rows,
    int const* expanded_dest_row_to_expanded_source_row, T* expanded_unpermuted_output, int64_t const cols,
    int64_t const num_rows, int64_t const k)
{
    // Load 128-bits per thread, according to the smallest data type we
    // read/write
    constexpr int64_t FINALIZE_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<T>::value;
    assert(cols % FINALIZE_ELEM_PER_THREAD == 0);

    int64_t const start_offset = threadIdx.x;
    int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
    int64_t const num_elems_in_col = cols / FINALIZE_ELEM_PER_THREAD;

    using DataElem = cutlass::Array<T, FINALIZE_ELEM_PER_THREAD>;

    int64_t const expanded_dest_row = blockIdx.x;
    int64_t const expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
    auto const* expanded_permuted_rows_v
        = reinterpret_cast<DataElem const*>(expanded_permuted_rows + expanded_dest_row * cols);
    auto* umpermuted_source_row_v
        = reinterpret_cast<DataElem*>(expanded_unpermuted_output + expanded_source_row * cols);

#pragma unroll
    for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        umpermuted_source_row_v[elem_index] = expanded_permuted_rows_v[elem_index];
    }
}

template <typename T>
void launchfinalizeScatterKernel(T const* expanded_permuted_rows, int const* expanded_dest_row_to_expanded_source_row,
    T* expanded_unpermuted_output, int64_t const cols, int64_t const num_rows, int64_t const k, cudaStream_t stream)
{
    int64_t const blocks = num_rows * k;
    int64_t const threads = FINALIZE_THREADS_PER_BLOCK;

    finalizeScatterKernel<T><<<blocks, threads, 0, stream>>>(expanded_permuted_rows,
        expanded_dest_row_to_expanded_source_row, expanded_unpermuted_output, cols, num_rows, k);
}

// ============================================================================
// MoeFCRunner Implementation
// ============================================================================
template <class T, class WeightType, class OutputType, class ScaleType, class Enable>
MoeFCRunner<T, WeightType, OutputType, ScaleType, Enable>::MoeFCRunner(
    int num_tokens, int hidden_size, int inter_size, int num_experts, int k)
{
    gemm_workspace_size_ = 0;
    int numBlocksPerSm = 0;
    // Number of threads my_kernel will be launched with
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, (void*) perTensorDynamicFp8QuantKernel<T, OutputType>, QUANTIZE_THREADS_PER_BLOCK, 0);
    per_token_grid_size_ = std::min(deviceProp.multiProcessorCount * numBlocksPerSm, num_tokens * k);
    problem_sizes_host_.resize(num_experts);
}

template <class T, class WeightType, class OutputType, class ScaleType, class Enable>
std::map<std::string, std::pair<size_t, size_t>>
MoeFCRunner<T, WeightType, OutputType, ScaleType, Enable>::getWorkspaceDeviceBufferSizes(
    int num_tokens, int hidden_size, int inter_size, int num_experts, int k, QuantMode quant_mode) const
{
    size_t num_moe_inputs = k * num_tokens;
    size_t const permuted_elems = num_moe_inputs * hidden_size;
    size_t const interbuf_elems = num_moe_inputs * inter_size;
    //  这里假设总是 glu like 激活
    size_t const glu_inter_elems = interbuf_elems * 2;

    size_t const cub_histogram_and_scan_temp_size
        = cubHistogramAndScan(nullptr, 0, nullptr, nullptr, nullptr, num_moe_inputs, num_experts);
    size_t const histogram_count_size = num_experts * sizeof(int);
    size_t const cumsum_counter_size = (num_experts + 1) * sizeof(int);
    size_t const expanded_dest_row_to_expanded_source_row_size = num_moe_inputs * sizeof(int);
    size_t const per_token_dest_row_start_indices_size = num_experts * sizeof(int);
    size_t const gathered_input_scales_size
        = num_moe_inputs * ((hidden_size + SFVecSizeK - 1) / SFVecSizeK) * sizeof(ScaleType);

    size_t const gathered_inputs_size = permuted_elems * sizeof(T);
    size_t const gemm1_output_size = glu_inter_elems * sizeof(OutputType);
    size_t const activation_output_size = interbuf_elems * sizeof(OutputType);
    size_t const quantized_activation_size = interbuf_elems * sizeof(T);

    size_t const gemm2_output_size = permuted_elems * sizeof(OutputType);
    size_t const dynamic_act_scale_size
        = num_moe_inputs * ((inter_size + SFVecSizeK - 1) / SFVecSizeK) * sizeof(ScaleType); //  暂且按照最大的情况分配

    // optional
    using PendingDequantType = float;
    size_t const gemm1_pending_dq_output_size = glu_inter_elems * sizeof(PendingDequantType);
    size_t const gemm2_pending_dq_output_size = permuted_elems * sizeof(PendingDequantType);
    size_t const alpha_scale_ptr_size = num_experts * sizeof(ScaleType) * 2;
    size_t const alpha_scale_ptr_array_size = num_experts * sizeof(ScaleType*) * 2;

    size_t const gemm_workspace_size = 16 * 1024 * 1024; // 16M should be enough for all cases

    size_t map_offset = 0;
    std::map<std::string, std::pair<size_t, size_t>> out_map;

#define ADD_NAME(name, size)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        auto aligned_size = align_size(size, CUDA_MEM_ALIGN);                                                          \
        out_map[#name] = std::pair{aligned_size, map_offset};                                                          \
        map_offset += aligned_size;                                                                                    \
    } while (false)
#define ADD(name) ADD_NAME(name, name##_size)

    ADD(cub_histogram_and_scan_temp);
    ADD(histogram_count);
    ADD(cumsum_counter);
    ADD(expanded_dest_row_to_expanded_source_row);
    ADD(per_token_dest_row_start_indices);
    ADD(gathered_input_scales);
    ADD(gathered_inputs);
    ADD(gemm1_output);
    ADD(activation_output);
    ADD(quantized_activation);
    ADD(gemm2_output);
    ADD(dynamic_act_scale);
    ADD(gemm1_pending_dq_output);
    ADD(gemm2_pending_dq_output);
    ADD_NAME(alpha_scale_ptr_array_fc1, alpha_scale_ptr_array_size);
    ADD_NAME(alpha_scale_ptr_array_fc2, alpha_scale_ptr_array_size);
    ADD_NAME(alpha_scale_ptr_fc1, alpha_scale_ptr_size);
    ADD_NAME(alpha_scale_ptr_fc2, alpha_scale_ptr_size);
    ADD(gemm_workspace);

    return out_map;
}

template <class T, class WeightType, class OutputType, class ScaleType, class Enable>
size_t MoeFCRunner<T, WeightType, OutputType, ScaleType, Enable>::getWorkspaceSize(
    int num_tokens, int hidden_size, int inter_size, int num_experts, int k, QuantMode quant_mode) const
{
    auto sizes_map = getWorkspaceDeviceBufferSizes(num_tokens, hidden_size, inter_size, num_experts, k, quant_mode);
    size_t workspace_size = std::accumulate(sizes_map.begin(), sizes_map.end(), size_t{0},
        [](size_t sum, auto const& pair) { return sum + pair.second.first; });

    return workspace_size;
}

template <class T, class WeightType, class OutputType, class ScaleType, class Enable>
void MoeFCRunner<T, WeightType, OutputType, ScaleType, Enable>::runMoeWithTorchDebug(void const* expert_inputs,
    void const* expert_weights1, void const* expert_weights2, int const* topk_indices, void const* expert_bias1,
    void const* expert_bias2, void* final_output, QuantMode quant_mode, QuantParams quant_params, int num_experts,
    int topk, int bs, int hidden_size, int inter_size, char* workspace, cudaStream_t stream,
    std::map<std::string, std::pair<void*, size_t>> intermediate_map)
{
    runMoe(expert_inputs, expert_weights1, expert_weights2, topk_indices, expert_bias1, expert_bias2, final_output,
        quant_mode, quant_params, num_experts, topk, bs, hidden_size, inter_size, workspace, stream);

    //   extractIntermediateResultsforTorch(workspace, num_experts, topk, bs,
    //                                      hidden_size, inter_size, quant_mode,
    //                                      intermediate_map, stream);
}

// Helper function to extract intermediate results from workspace
template <class T, class WeightType, class OutputType, class ScaleType, class Enable>
void MoeFCRunner<T, WeightType, OutputType, ScaleType, Enable>::extractIntermediateResultsforTorch(char* workspace,
    int num_experts, int topk, int bs, int hidden_size, int inter_size, QuantMode quant_mode,
    std::map<std::string, std::pair<void*, size_t>> intermediate_map, cudaStream_t stream)
{
    char* workspace_ptr = workspace;
    int total_tokens = bs * topk;

    auto workspaces = getWorkspaceDeviceBufferSizes(bs, hidden_size, inter_size, num_experts, topk, quant_mode);
    auto getWsPtr = [&](auto type, std::string const& name)
    {
        return workspaces.at(name).first ? reinterpret_cast<decltype(type)*>(workspace_ptr + workspaces.at(name).second)
                                         : nullptr;
    };
    auto getPtrSize = [&](std::string const& name)
    { return workspaces.at(name).first ? reinterpret_cast<size_t>(workspaces.at(name).first) : 0; };

    auto* gathered_inputs = getWsPtr(T{}, "gathered_inputs");
    auto* gathered_input_scales = getWsPtr(ScaleType{}, "gathered_input_scales");
    OutputType* gemm1_output = getWsPtr(OutputType{}, "gemm1_output");
    OutputType* activation_output = getWsPtr(OutputType{}, "activation_output");
    OutputType* gemm2_output = getWsPtr(OutputType{}, "gemm2_output");

    auto gemm1_output_size = getPtrSize("gemm1_output");
    auto gemm2_output_size = getPtrSize("gemm2_output");

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <class T, class WeightType, class OutputType, class ScaleType, class Enable>
void MoeFCRunner<T, WeightType, OutputType, ScaleType, Enable>::runMoe(void const* expert_inputs,
    void const* expert_weights1, void const* expert_weights2, int const* topk_indices, void const* expert_bias1,
    void const* expert_bias2, void* final_output, QuantMode quant_mode, QuantParams quant_param, int num_experts,
    int topk, int num_tokens, int hidden_size, int inter_size, char* workspace, cudaStream_t stream)
{
    // Cast inputs to proper types
    T const* inputs = static_cast<T const*>(expert_inputs);
    T const* weights1 = static_cast<T const*>(expert_weights1);
    T const* weights2 = static_cast<T const*>(expert_weights2);

    int num_elements = num_tokens * topk;
    char* workspace_ptr = workspace;
    auto workspaces = getWorkspaceDeviceBufferSizes(num_tokens, hidden_size, inter_size, num_experts, topk, quant_mode);
    auto getWsPtr = [&](auto type, std::string const& name)
    {
        return workspaces.at(name).first ? reinterpret_cast<decltype(type)*>(workspace_ptr + workspaces.at(name).second)
                                         : nullptr;
    };

    void* d_temp_storage = (void*) (workspace_ptr + workspaces.at("cub_histogram_and_scan_temp").second);
    auto d_temp_storage_size = workspaces.at("cub_histogram_and_scan_temp").first;

    int* d_counts = getWsPtr(int{}, "histogram_count");
    int* d_cumsum_counter = getWsPtr(int{}, "cumsum_counter");
    int* expanded_dest_row_to_expanded_source_row = getWsPtr(int{}, "expanded_dest_row_to_expanded_source_row");
    int* per_token_dest_row_start_indices = getWsPtr(int{}, "per_token_dest_row_start_indices");
    T* gathered_inputs = getWsPtr(T{}, "gathered_inputs");
    OutputType* gemm1_output = getWsPtr(OutputType{}, "gemm1_output");
    OutputType* activation_output = getWsPtr(OutputType{}, "activation_output");
    T* quantized_activation = getWsPtr(T{}, "quantized_activation");
    OutputType* gemm2_output = getWsPtr(OutputType{}, "gemm2_output");
    ScaleType* d_dynamic_scale_workspace = getWsPtr(ScaleType{}, "dynamic_act_scale");
    ScaleType* gathered_input_scales = getWsPtr(ScaleType{}, "gathered_input_scales");

    float* gemm1_pending_dq_output = getWsPtr(float{}, "gemm1_pending_dq_output");
    float* gemm2_pending_dq_output = getWsPtr(float{}, "gemm2_pending_dq_output");

    char* gemm_workspace = getWsPtr(char{}, "gemm_workspace");
    size_t gemm_workspace_size = workspaces.at("gemm_workspace").first;

    // Step 1: Compute histogram
    // Step 2: Compute cumulative sum
    cubHistogramAndScan(d_temp_storage, d_temp_storage_size, topk_indices, d_counts, d_cumsum_counter, num_elements,
        num_experts, stream);
    // Step 3: Gather inputs by expert
    cudaMemsetAsync(per_token_dest_row_start_indices, 0, num_experts * sizeof(int), stream);
    launchGetGroupedIndex(topk_indices, d_cumsum_counter, expanded_dest_row_to_expanded_source_row, num_elements, topk,
        per_token_dest_row_start_indices, stream);
    launchExpandInputRowsGatherKernel(
        inputs, expanded_dest_row_to_expanded_source_row, gathered_inputs, num_tokens, hidden_size, topk, stream);
    CUDA_CHECK(cudaGetLastError());

    if constexpr (use_fp8)
    {
        if (quant_mode.hasFp8BlockWise())
        {
            launchExpandInputScaleBlockGatherKernel(quant_param.fp8.act_scales,
                expanded_dest_row_to_expanded_source_row, gathered_input_scales, num_tokens,
                (hidden_size + SFVecSizeK - 1) / SFVecSizeK, topk, stream);
        }
        else if (quant_mode.hasFp8RowWise())
        {
            launchExpandInputScaleGatherKernel(quant_param.fp8.act_scales, expanded_dest_row_to_expanded_source_row,
                gathered_input_scales, num_tokens, topk, stream);
        }
        else if (quant_mode.hasFp8Qdq())
        {
            gathered_input_scales = (ScaleType*) quant_param.fp8.act_scales;
        }
    }
    else
    {
        gathered_input_scales = (ScaleType*) nullptr;
    }

    // CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    // Step 4: First GEMM
    Gemm(gathered_inputs, weights1, d_counts, d_cumsum_counter, gemm1_output, gemm1_pending_dq_output, quant_mode,
        gathered_input_scales, quant_param.fp8.weight1_scales, num_experts, num_tokens, topk, inter_size * 2,
        hidden_size, gemm_workspace, gemm_workspace_size, "GEMM1", stream);

    // Step 5: Gated activation
    launchGatedActivation(gemm1_output, activation_output, num_elements, inter_size, stream);

    if constexpr (use_fp8)
    {
        // Step 6: Dynamic quantization
        if (quant_mode.hasFp8BlockWise())
        {
            launchBlockDynamicFp8Quant(
                activation_output, quantized_activation, d_dynamic_scale_workspace, num_elements, inter_size, stream);
        }
        else if (quant_mode.hasFp8RowWise())
        {
            launchTokenDynamicFp8Quant(
                activation_output, quantized_activation, d_dynamic_scale_workspace, num_elements, inter_size, stream);
        }
        else if (quant_mode.hasFp8Qdq())
        {
            launchTensorDynamicFp8Quant(activation_output, d_dynamic_scale_workspace, quantized_activation, num_experts,
                inter_size, num_elements, per_token_grid_size_, stream);
        }
    }
    else
    {
        quantized_activation = (T*) activation_output;
        d_dynamic_scale_workspace = (ScaleType*) nullptr;
    }

    // Step 7: Second GEMM
    Gemm(quantized_activation, weights2, d_counts, d_cumsum_counter, gemm2_output, gemm2_pending_dq_output, quant_mode,
        d_dynamic_scale_workspace, quant_param.fp8.weight2_scales, num_experts, num_tokens, topk, hidden_size,
        inter_size, gemm_workspace, gemm_workspace_size, "GEMM2", stream);

    // CUDA_CHECK(cudaStreamSynchronize(stream));
    // Step 8: Scatter the expert-grouped output back to original token
    // order
    launchfinalizeScatterKernel(gemm2_output, expanded_dest_row_to_expanded_source_row,
        static_cast<OutputType*>(final_output), hidden_size, num_tokens, topk, stream);

    // CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <class T, class WeightType, class OutputType, class ScaleType, class Enable>
void MoeFCRunner<T, WeightType, OutputType, ScaleType, Enable>::Gemm(T const* expert_inputs,
    WeightType const* expert_weights, int const* d_counts, int const* d_cumsum_counter, OutputType* gemm_output,
    float* pending_dq_output, QuantMode const& quant_mode, ScaleType const* scale_a, ScaleType const* scale_b,
    int num_experts, int num_tokens, int topk, int N, int K, char* workspace, size_t workspace_size,
    std::string operation_name, cudaStream_t stream)
{
#define RUN_GEMM_WITH_QUANTMODE(QUANT_MODE)                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        GroupedGemmInput<T, WeightType, OutputType, ScaleType, QuantMode::QUANT_MODE> gemm_input{.A = expert_inputs,   \
            .B = expert_weights,                                                                                       \
            .scales_a = scale_a,                                                                                       \
            .scales_b = scale_b,                                                                                       \
            .C = nullptr /*expert_bias*/,                                                                              \
            .D = gemm_output,                                                                                          \
            .cumsum_counter = d_cumsum_counter,                                                                        \
            .pending_dq_output = pending_dq_output,                                                                    \
            .quant_mode = quant_mode,                                                                                  \
            .num_rows = num_tokens,                                                                                    \
            .topk = topk,                                                                                              \
            .N = N,                                                                                                    \
            .K = K,                                                                                                    \
            .num_experts = num_experts,                                                                                \
            .workspace_size = workspace_size,                                                                          \
            .workspace_ptr = workspace,                                                                                \
            .sm = moe_gemm_runner_.getSM(),                                                                            \
            .operation_name = operation_name,                                                                          \
            .stream = stream};                                                                                         \
        moe_gemm_runner_.runGemm(gemm_input);                                                                          \
    } while (0)

    if constexpr (use_fp8)
    {
        if (quant_mode.hasFp8BlockWise())
        {
            RUN_GEMM_WITH_QUANTMODE(FP8_BLOCKWISE);
        }
        else if (quant_mode.hasFp8RowWise())
        {
            RUN_GEMM_WITH_QUANTMODE(FP8_ROWWISE);
        }
        else if (quant_mode.hasFp8Qdq())
        {
            RUN_GEMM_WITH_QUANTMODE(FP8_QDQ);
        }
    }
    else
    {
        RUN_GEMM_WITH_QUANTMODE(NONE);
    }
#undef RUN_GEMM_WITH_QUANTMODE
}