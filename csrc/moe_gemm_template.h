#pragma once
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <type_traits>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"
#include "default_gemm_grouped_with_visitor.h"
#include "gemm_grouped_with_visitor.h"
#include "quantization.h"
#include "util.h"

using namespace cute;

template <typename AType, typename BType, typename OType, typename ScaleType = float,
    typename QuantModeType = QuantMode::NONE>
struct GroupedGemmInput
{
    AType const* A = nullptr;
    BType const* B = nullptr;
    ScaleType const* scales_a = nullptr;
    ScaleType const* scales_b = nullptr;
    OType const* C = nullptr;
    OType* D = nullptr;
    int const* cumsum_counter = nullptr;

    float* pending_dq_output = nullptr;

    QuantMode quant_mode{0};

    int64_t num_rows = 0;
    int64_t topk = 0;
    int64_t N = 0;
    int64_t K = 0;
    int num_experts = 0;
    size_t workspace_size = 0;
    char* workspace_ptr = nullptr;
    int sm = 80;
    std::string operation_name;
    cudaStream_t stream = 0;
    bool fused = true;
};

template <typename ElementA, typename ElementB, typename ElementD, typename ElementBlockScale, typename StrideA,
    typename StrideB, typename StrideC, typename StrideD, typename LayoutSFA, typename LayoutSFB, typename ProblemShape>
__global__ void configureTmaWarpSpecializeBlockWiseGemmKernel(int num_experts, int const* cumsum_counter,
    ElementA const* block_A, ElementBlockScale const* blockscale_block_A, ElementB const* block_B,
    ElementBlockScale const* blockscale_block_B, ElementD* block_D, int K, int N, ElementA const** ptr_A,
    ElementB const** ptr_B, ElementD** ptr_D, ElementBlockScale const** ptr_blockscale_A,
    ElementBlockScale const** ptr_blockscale_B, StrideA* strides_A, StrideB* strides_B, StrideC* strides_C,
    StrideD* strides_D, LayoutSFA* layout_SFA, LayoutSFB* layout_SFB, ProblemShape* problem_sizes_device)
{
    constexpr int ScaleGranularityM = 1;
    constexpr int ScaleGranularityN = 128;
    constexpr int ScaleGranularityK = 128;
    using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
        ScaleGranularityK, cute::GMMA::Major::K, cute::GMMA::Major::K>;
    int expert_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert_id < num_experts)
    {
        int start_idx = cumsum_counter[expert_id];
        int end_idx = cumsum_counter[expert_id + 1];
        int M = end_idx - start_idx;
        auto group_layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
        auto group_layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

        ptr_A[expert_id] = block_A + start_idx * K;
        ptr_B[expert_id] = block_B + expert_id * K * N;
        ptr_D[expert_id] = block_D + start_idx * N;
        ptr_blockscale_A[expert_id] = blockscale_block_A
            + size(filter_zeros(ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(start_idx, N, K, 1))));
        ptr_blockscale_B[expert_id] = blockscale_block_B + expert_id * size(filter_zeros(group_layout_SFB));

        strides_A[expert_id] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        strides_B[expert_id] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        strides_C[expert_id] = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
        strides_D[expert_id] = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
        layout_SFA[expert_id] = group_layout_SFA;
        layout_SFB[expert_id] = group_layout_SFB;

        problem_sizes_device[expert_id] = ProblemShape({M, N, K});
    }
}

struct PingpongConfig
{
    using ElementA = cutlass::float_e4m3_t;
    using TileShape = cute::Shape<cute::_64, cute::_128, cute::_128>;
    using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8BlockScaledAccum;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
};

  struct CooperativeConfig {
    using ElementA = cutlass::float_e4m3_t;
    using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
    using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  };


template <typename T, typename WeightType, typename GemmOutputType, typename ScaleType, typename arch,
    typename QuantModeType, typename MmaConfig,
    typename std::enable_if_t<std::is_same_v<T, __nv_fp8_e4m3> && std::is_same_v<WeightType, T>
        && std::is_same_v<QuantModeType, QuantMode::FP8_BLOCKWISE>>* = nullptr>
struct moeGemmTmaWarpSpecializedBlockWise
{
    static_assert(90 <= arch::kMinComputeCapability, "moeGemmTmaWarpSpecializedBlockWise requires SM90+.");
    using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>; // <M,N,K>
                                                                                       // per group

    using ElementA = typename NvToCutlassTypeAdapter<T>::type; // Element type for
                                                               // A matrix operand
    using LayoutA = cutlass::layout::RowMajor;                 // Layout type for A matrix operand
    constexpr static int AlignmentA
        = 128 / cutlass::sizeof_bits<ElementA>::value; // Memory access granularity/alignment of A
                                                       // matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using ElementB = typename NvToCutlassTypeAdapter<WeightType>::type; // Element type for
                                                                        // B matrix operand
    using LayoutB = cutlass::layout::ColumnMajor;                       // Layout type for B matrix operand
    constexpr static int AlignmentB
        = 128 / cutlass::sizeof_bits<ElementB>::value; // Memory access granularity/alignment of B
                                                       // matrix in units of elements (up to 16 bytes)

    // C matrix configuration
    using ElementC = typename NvToCutlassTypeAdapter<GemmOutputType>::type; // Element type for C and D matrix operands
    using LayoutC = cutlass::layout::RowMajor;                              // Layout type for C and D matrix operands
    constexpr static int AlignmentC
        = 128 / cutlass::sizeof_bits<ElementC>::value; // Memory access granularity/alignment of C
                                                       // matrix in units of elements (up to 16 bytes)

    // D matrix configuration
    using ElementD = ElementC;
    using LayoutD = LayoutC;
    constexpr static int AlignmentD = AlignmentC;

    // Core kernel configurations
    using ElementAccumulator = float; // Element type for internal accumulation
    using ElementBlockScale = float;  // Element type for blockscaling during accumulation
    using ElementCompute = float;     // Element type for epilogue computation

    using ArchTag = cutlass::arch::Sm90;                  // Tag indicating the minimum SM that
                                                          // supports the intended feature
    using OperatorClass = cutlass::arch::OpClassTensorOp; // Operator class tag
    // using TileShape = cute::Shape<cute::_128, cute::_128,
    //     cute::_128>; // Threadblock-level tile size
    // using ClusterShape = cute::Shape<cute::_1, cute::_2,
    //     cute::_1>; // Shape of the threadblocks in a cluster

    using TileShape = typename MmaConfig::TileShape;           // Threadblock-level tile size
    using ClusterShape = typename MmaConfig::ClusterShape;     // Shape of the threadblocks in a cluster

    constexpr static int ScaleGranularityM = 1;
    constexpr static int ScaleGranularityN = 128;
    constexpr static int ScaleGranularityK = 128;

    using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
        ScaleGranularityK, cute::GMMA::Major::K, cute::GMMA::Major::K>;

    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA()); // Layout type for SFA
                                                                 // matrix operand
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB()); // Layout type for SFB
                                                                 // matrix operand

    using KernelSchedule = typename MmaConfig::KernelSchedule;
    using EpilogueSchedule = typename MmaConfig::EpilogueSchedule;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
    // using FusionOperation = cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>;
    static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
    using CustomEVTIdentity =  // acc
        cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::Sm90Compute<
            cutlass::epilogue::thread::Identity, ElementD, ElementAccumulator, RoundStyle>,
        cutlass::epilogue::fusion::Sm90AccFetch
        >;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag, OperatorClass,
        TileShape, ClusterShape, EpilogueTileType, ElementAccumulator, ElementCompute, ElementC, LayoutC*, AlignmentC,
        ElementD, LayoutD*, AlignmentD, EpilogueSchedule, CustomEVTIdentity>::CollectiveOp;

    using CollectiveMainloopWithGroupWiseScaling = typename cutlass::gemm::collective::CollectiveBuilder<ArchTag,
        OperatorClass, ElementA, cute::tuple<LayoutA*, LayoutSFA*>, AlignmentA, ElementB,
        cute::tuple<LayoutB*, LayoutSFB*>, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloopWithGroupWiseScaling,
        CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Extract information from Gemm kernel.
    using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;
    using ElementScalar = typename EpilogueOutputOp::ElementScalar;

    using StrideA = typename Gemm::GemmKernel::InternalStrideA;
    using StrideB = typename Gemm::GemmKernel::InternalStrideB;
    using StrideC = typename Gemm::GemmKernel::InternalStrideC;
    using StrideD = typename Gemm::GemmKernel::InternalStrideD;
    using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90GroupParams<
        cute::Shape<int, int, int>>::RasterOrderOptions;
    static_assert(cute::is_same_v<ElementAccumulator, ElementBlockScale>,
        "ElementAccumulator and ElementBlockScale should be same datatype");

    static void call(GroupedGemmInput<T, WeightType, GemmOutputType, ScaleType, QuantModeType> inputs)
    {
        static cutlass::KernelHardwareInfo kernel_hw_info
            = cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename Gemm::GemmKernel>(0 /*device_id*/);

        auto block_A = (ElementA const*) inputs.A;
        auto block_B = (ElementB const*) inputs.B;
        // auto block_C = (ElementC*)inputs.C;
        auto block_D = (ElementD*) inputs.D;
        auto blockscale_block_A = (ElementBlockScale const*) inputs.scales_a;
        auto blockscale_block_B = (ElementBlockScale const*) inputs.scales_b;
        auto ptr_A = (ElementA const**) (inputs.workspace_ptr);
        auto ptr_B = (ElementB const**) (ptr_A + inputs.num_experts);
        auto ptr_C = (ElementC const**) (ptr_B + inputs.num_experts);
        auto ptr_D = (ElementD**) (ptr_C + inputs.num_experts);
        auto ptr_blockscale_A = (ElementBlockScale const**) (ptr_D + inputs.num_experts);
        auto ptr_blockscale_B = (ElementBlockScale const**) (ptr_blockscale_A + inputs.num_experts);
        auto stride_A = (StrideA*) (align_pointer(ptr_blockscale_B + inputs.num_experts));
        auto stride_B = (StrideB*) (stride_A + inputs.num_experts);
        auto stride_C = (StrideC*) (stride_B + inputs.num_experts);
        auto stride_D = (StrideD*) (stride_C + inputs.num_experts);
        auto layout_SFA = (LayoutSFA*) (stride_D + inputs.num_experts);
        auto layout_SFB = (LayoutSFB*) (layout_SFA + inputs.num_experts);

        auto* problem_sizes
            = (typename ProblemShape::UnderlyingProblemShape*) (align_pointer(layout_SFB + inputs.num_experts));

        auto gemm_workspace = (uint8_t*) (align_pointer(problem_sizes + inputs.num_experts));

        constexpr int BLOCK_SIZE = 128;
        auto GRID_SIZE = (inputs.num_experts + BLOCK_SIZE - 1) / BLOCK_SIZE;
        configureTmaWarpSpecializeBlockWiseGemmKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE), 0, inputs.stream>>>(
            inputs.num_experts, inputs.cumsum_counter, block_A, blockscale_block_A, block_B, blockscale_block_B,
            block_D, inputs.K, inputs.N, ptr_A, ptr_B, ptr_D, ptr_blockscale_A, ptr_blockscale_B, stride_A, stride_B,
            stride_C, stride_D, layout_SFA, layout_SFB, problem_sizes);

        typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
            {inputs.num_experts, problem_sizes, (typename ProblemShape::UnderlyingProblemShape*) nullptr},
            {ptr_A, stride_A, ptr_B, stride_B, ptr_blockscale_A, layout_SFA, ptr_blockscale_B, layout_SFB},
            {{}, // epilogue.thread
                nullptr, stride_C, ptr_D, stride_D},
            kernel_hw_info};

        // auto& fusion_args = arguments.epilogue.thread;

        // fusion_args.alpha = 1.0;
        // fusion_args.beta = 0.0;
        // fusion_args.alpha_ptr = nullptr;
        // fusion_args.beta_ptr = nullptr;
        // fusion_args.alpha_ptr_array = nullptr;
        // fusion_args.beta_ptr_array = nullptr;
        // // Single alpha and beta for all groups
        // fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
        // fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

        // arguments.scheduler.raster_order = RasterOrderOptions::Heuristic;
        // The tile scheduler will swizzle up to 8 and with the nearest multiple
        // of 2 (i.e., 1, 2, 4, and 8)
        // arguments.scheduler.max_swizzle_size = 1;
        Gemm gemm;

        // Allocate workspace memory
        // 经验表明，需要 84480 byte workspace

        // Check if the problem size is supported or not
        CUTLASS_CHECK(gemm.can_implement(arguments));

        // Initialize CUTLASS kernel with arguments and workspace pointer
        CUTLASS_CHECK(gemm.initialize(arguments, gemm_workspace, inputs.stream));
        CUTLASS_CHECK(gemm.run(inputs.stream));
    }
};

template <typename ElementA, typename ElementB, typename ElementD, typename StrideA, typename StrideB, typename StrideC,
    typename StrideD, typename ProblemShape>
__global__ void configureTmaWarpSpecializeGemmKernel(int num_experts, int const* cumsum_counter,
    ElementA const* block_A, ElementB const* block_B, ElementD* block_D, int K, int N, ElementA const** ptr_A,
    ElementB const** ptr_B, ElementD** ptr_D, StrideA* strides_A, StrideB* strides_B, StrideC* strides_C,
    StrideD* strides_D, ProblemShape* problem_sizes_device)
{
    int expert_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert_id < num_experts)
    {
        int start_idx = cumsum_counter[expert_id];
        int end_idx = cumsum_counter[expert_id + 1];
        int M = end_idx - start_idx;

        ptr_A[expert_id] = block_A + start_idx * K;
        ptr_B[expert_id] = block_B + expert_id * K * N;
        ptr_D[expert_id] = block_D + start_idx * N;

        strides_A[expert_id] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        strides_B[expert_id] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        strides_C[expert_id] = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
        strides_D[expert_id] = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
        problem_sizes_device[expert_id] = ProblemShape({M, N, K});
    }
}

template <typename T, typename WeightType, typename GemmOutputType, typename ScaleType, typename arch,
    typename QuantModeType,
    typename std::enable_if_t<!std::is_same_v<T, __nv_fp8_e4m3>
        && !std::is_same_v<WeightType, __nv_fp8_e4m3>>* = nullptr>
struct moeGemmTmaWarpSpecialized
{
    static_assert(90 <= arch::kMinComputeCapability, "moeGemmTmaWarpSpecialized requires SM90+.");
    using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>; // <M,N,K>
                                                                                       // per group

    using ElementA = typename NvToCutlassTypeAdapter<T>::type; // Element type for
                                                               // A matrix operand
    using LayoutA = cutlass::layout::RowMajor;                 // Layout type for A matrix operand
    constexpr static int AlignmentA
        = 128 / cutlass::sizeof_bits<ElementA>::value; // Memory access granularity/alignment of A
                                                       // matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using ElementB = typename NvToCutlassTypeAdapter<WeightType>::type; // Element type for
                                                                        // B matrix operand
    using LayoutB = cutlass::layout::ColumnMajor;                       // Layout type for B matrix operand
    constexpr static int AlignmentB
        = 128 / cutlass::sizeof_bits<ElementB>::value; // Memory access granularity/alignment of B
                                                       // matrix in units of elements (up to 16 bytes)

    // C matrix configuration
    using ElementC = typename NvToCutlassTypeAdapter<GemmOutputType>::type; // Element type for C and D matrix operands
    using LayoutC = cutlass::layout::RowMajor;                              // Layout type for C and D matrix operands
    constexpr static int AlignmentC
        = 128 / cutlass::sizeof_bits<ElementC>::value; // Memory access granularity/alignment of C
                                                       // matrix in units of elements (up to 16 bytes)

    // D matrix configuration
    using ElementD = ElementC;
    using LayoutD = LayoutC;
    constexpr static int AlignmentD = AlignmentC;

    // Core kernel configurations
    using ElementAccumulator = float;    // Element type for internal accumulation
    using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that supports the intended feature
    using OperatorClass = cutlass::arch::OpClassTensorOp;             // Operator class tag
    using StageCountType = cutlass::gemm::collective::StageCountAuto; // Stage count maximized based on the tile size

    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
    using TileShape = cute::Shape<cute::_256, cute::_128, cute::_64>;
    using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp, TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator, ElementC, LayoutC*, AlignmentC, ElementC, LayoutC*, AlignmentC,
        EpilogueSchedule, cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<ArchTag, OperatorClass, ElementA,
        LayoutA*, AlignmentA, ElementB, LayoutB*, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename Gemm::GemmKernel::InternalStrideA;
    using StrideB = typename Gemm::GemmKernel::InternalStrideB;
    using StrideC = typename Gemm::GemmKernel::InternalStrideC;
    using StrideD = typename Gemm::GemmKernel::InternalStrideD;

    static void call(GroupedGemmInput<T, WeightType, GemmOutputType, ScaleType, QuantModeType> inputs)
    {
        static cutlass::KernelHardwareInfo kernel_hw_info
            = cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename Gemm::GemmKernel>(0 /*device_id*/);

        auto block_A = (ElementA const*) inputs.A;
        auto block_B = (ElementB const*) inputs.B;
        auto block_C = (ElementC const*) inputs.C;
        auto block_D = (ElementD*) inputs.D;
        auto ptr_A = (ElementA const**) (inputs.workspace_ptr);
        auto ptr_B = (ElementB const**) (ptr_A + inputs.num_experts);
        auto ptr_C = (ElementC const**) (ptr_B + inputs.num_experts);
        auto ptr_D = (ElementD**) (ptr_C + inputs.num_experts);
        auto stride_A = (StrideA*) (align_pointer(ptr_D + inputs.num_experts));
        auto stride_B = (StrideB*) (stride_A + inputs.num_experts);
        auto stride_C = (StrideC*) (stride_B + inputs.num_experts);
        auto stride_D = (StrideD*) (stride_C + inputs.num_experts);

        auto* problem_sizes
            = (typename ProblemShape::UnderlyingProblemShape*) (align_pointer(stride_D + inputs.num_experts));

        auto gemm_workspace = (uint8_t*) (align_pointer(problem_sizes + inputs.num_experts));

        constexpr int BLOCK_SIZE = 128;
        auto GRID_SIZE = (inputs.num_experts + BLOCK_SIZE - 1) / BLOCK_SIZE;
        configureTmaWarpSpecializeGemmKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE), 0, inputs.stream>>>(
            inputs.num_experts, inputs.cumsum_counter, block_A, block_B, block_D, inputs.K, inputs.N, ptr_A, ptr_B,
            ptr_D, stride_A, stride_B, stride_C, stride_D, problem_sizes);

        typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
            {inputs.num_experts, problem_sizes, (typename ProblemShape::UnderlyingProblemShape*) nullptr},
            {ptr_A, stride_A, ptr_B, stride_B},
            {{}, // epilogue.thread
                ptr_C, stride_C, ptr_D, stride_D},
            kernel_hw_info};

        auto& fusion_args = arguments.epilogue.thread;

        fusion_args.alpha = 1.0;
        fusion_args.beta = 0.0;
        fusion_args.alpha_ptr = nullptr;
        fusion_args.beta_ptr = nullptr;
        fusion_args.alpha_ptr_array = nullptr;
        fusion_args.beta_ptr_array = nullptr;
        // Single alpha and beta for all groups
        fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
        fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

        Gemm gemm;

        // Allocate workspace memory
        // 经验表明，需要 84480 byte workspace

        // Check if the problem size is supported or not
        CUTLASS_CHECK(gemm.can_implement(arguments));

        // Initialize CUTLASS kernel with arguments and workspace pointer
        CUTLASS_CHECK(gemm.initialize(arguments, gemm_workspace, inputs.stream));
        CUTLASS_CHECK(gemm.run(inputs.stream));
    }
};

template <typename ElementA, typename ElementB, typename ElementD, typename LeadingDimElement, typename ProblemShape>
__global__ void configureGemmKernel(int num_experts, int const* cumsum_counter, ElementA* A, ElementB* B, ElementD* D,
    int N, int K, ElementA** ptr_A, ElementB** ptr_B, ElementD** ptr_D, LeadingDimElement* lda, LeadingDimElement* ldb,
    LeadingDimElement* ldd, ProblemShape* problem_sizes_device)
{
    int expert_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert_id < num_experts)
    {
        int start_idx = cumsum_counter[expert_id];
        int end_idx = cumsum_counter[expert_id + 1];
        int M = end_idx - start_idx;

        ptr_A[expert_id] = A + start_idx * K;
        ptr_B[expert_id] = B + expert_id * K * N;
        ptr_D[expert_id] = D + start_idx * N;

        lda[expert_id] = K;
        // NOTE ColumnMajor
        ldb[expert_id] = K;
        ldd[expert_id] = N;
        problem_sizes_device[expert_id] = ProblemShape(M, N, K);
    }
}

template <typename T, typename WeightType, typename GemmOutputType, typename ScaleType, typename arch,
    typename QuantModeType,
    typename std::enable_if_t<((!std::is_same_v<T, __nv_fp8_e4m3>) && (!std::is_same_v<T, __nv_fp8_e5m2>) )
        && std::is_same_v<T, WeightType>>* = nullptr>
struct moeGemm
{
    using ElementA = typename NvToCutlassTypeAdapter<T>::type; // Element type for
                                                               // A matrix operand
    using LayoutA = cutlass::layout::RowMajor;                 // Layout type for A matrix operand
    constexpr static int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    using ElementB = typename NvToCutlassTypeAdapter<WeightType>::type; // Element type for
                                                                        // B matrix operand
    using LayoutB = cutlass::layout::ColumnMajor;
    constexpr static int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value; // Memory access
                                                                                   // granularity/alignment of B
    // C matrix configuration
    using ElementC = typename NvToCutlassTypeAdapter<GemmOutputType>::type; // Element type for C and D matrix operands
    using LayoutC = cutlass::layout::RowMajor;                              // Layout type for C and D matrix operands
    constexpr static int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value; // Memory access
                                                                                   // granularity/alignment of C

    // D matrix configuration
    using ElementD = ElementC;
    using LayoutD = LayoutC;
    constexpr static int AlignmentD = AlignmentC;

    using LeadingDimElement = int64_t;

    using GemmKernel =
        typename cutlass::gemm::kernel::DefaultGemmGrouped<ElementA, LayoutA, cutlass::ComplexTransform::kNone,
            AlignmentA, ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentD, ElementD, LayoutD, float,
            cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>,
            cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
            cutlass::epilogue::thread::LinearCombination<ElementD, 128 / cutlass::sizeof_bits<ElementD>::value, float,
                float>,
            // NOTE: Threadblock swizzling is currently not supported by CUTLASS's
            // grouped kernels. This parameter is passed in at present to match the
            // APIs of other kernels. The parameter is unused within the kernel.
            cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
            // NOTE: Cutlass grouped gemm float 仅支持 stage = 2
            sizeof(std::conditional_t<std::is_same_v<T, float>, int16_t, int32_t>)>::GemmKernel;
    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    static void call(GroupedGemmInput<T, WeightType, GemmOutputType, ScaleType, QuantModeType>& inputs)
    {
        auto block_A = (ElementA*) inputs.A;
        auto block_B = (ElementB*) inputs.B;
        // auto block_C = (ElementC*)inputs.C;
        auto block_D = (ElementD*) inputs.D;

        auto ptr_A = (ElementA**) (inputs.workspace_ptr);
        auto ptr_B = (ElementB**) (ptr_A + inputs.num_experts);
        auto ptr_C = (ElementC**) (ptr_B + inputs.num_experts);
        auto ptr_D = (ElementD**) (ptr_C + inputs.num_experts);

        auto lda = (LeadingDimElement*) (align_pointer(ptr_D + inputs.num_experts));
        auto ldb = (LeadingDimElement*) (align_pointer(lda + inputs.num_experts));
        auto ldc = (LeadingDimElement*) (align_pointer(ldb + inputs.num_experts));
        auto ldd = (LeadingDimElement*) (align_pointer(ldc + inputs.num_experts));

        auto problem_size_workspace = (cutlass::gemm::GemmCoord*) (align_pointer(ldd + inputs.num_experts));
        auto gemm_workspace = (uint8_t*) (align_pointer(problem_size_workspace + inputs.num_experts));

        constexpr int BLOCK_SIZE = 128;
        auto GRID_SIZE = (inputs.num_experts + BLOCK_SIZE - 1) / BLOCK_SIZE;
        configureGemmKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE), 0, inputs.stream>>>(inputs.num_experts,
            inputs.cumsum_counter, block_A, block_B, block_D, inputs.N, inputs.K, ptr_A, ptr_B, ptr_D, lda, ldb, ldd,
            problem_size_workspace);

        GemmGrouped gemm;
        typename GemmGrouped::EpilogueOutputOp::Params epilogue_op(1.0f, 0.0f);
        int threadblock_count = GemmGrouped::sufficient(nullptr, inputs.num_experts);

        typename GemmGrouped::Arguments arguments(problem_size_workspace, inputs.num_experts, threadblock_count,
            epilogue_op, ptr_A, ptr_B, ptr_C, ptr_D, lda, ldb, ldc, ldd, (cutlass::gemm::GemmCoord*) nullptr);
        // Check if the problem size is supported or not
        CUTLASS_CHECK(gemm.can_implement(arguments));

        // Initialize CUTLASS kernel with arguments and workspace pointer
        CUTLASS_CHECK(gemm.initialize(arguments, gemm_workspace, inputs.stream));
        CUTLASS_CHECK(gemm.run(inputs.stream));
    }
};

__global__ void combimeScalesAndsetupAlphaPtrArrayKernel(
    float const* scale1, float const* scale2, float* combined_scale, float** alpha_ptr_array, int num_experts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_experts)
    {
        combined_scale[idx] = scale1[0] * scale2[idx]; // scale1 is per-tensor, scale2 is per-expert
        alpha_ptr_array[idx] = &combined_scale[idx];
    }
}

template <typename T, typename WeightType, typename GemmOutputType, typename ScaleType, typename arch,
    typename QuantModeType,
    typename std::enable_if_t<std::is_same_v<T, __nv_fp8_e4m3> && std::is_same_v<WeightType, T>
        && std::is_same_v<QuantModeType, QuantMode::FP8_QDQ>>* = nullptr>
struct moeGemmTensorWise
{
    using ElementA = typename NvToCutlassTypeAdapter<T>::type;              // Element type for
                                                                            // A matrix operand
    using ElementB = typename NvToCutlassTypeAdapter<WeightType>::type;     // Element type for
                                                                            // B matrix operand
    using ElementC = typename NvToCutlassTypeAdapter<GemmOutputType>::type; // Element type for C and D matrix operands
    // D matrix configuration
    using ElementD = ElementC;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = LayoutC;

    constexpr static int ElementsPerAccessA = 128 / cutlass::sizeof_bits<ElementA>::value;
    constexpr static int ElementsPerAccessB = 128 / cutlass::sizeof_bits<ElementB>::value;
    using LeadingDimElement = int64_t;
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGroupedPerGroupScale<ElementA, LayoutA,
        cutlass::ComplexTransform::kNone, ElementsPerAccessA, ElementB, LayoutB, cutlass::ComplexTransform::kNone,
        ElementsPerAccessB, ElementD, LayoutD, ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
        cutlass::gemm::GemmShape<64, 128, 64>, cutlass::gemm::GemmShape<64, 32, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombination<ElementD, 128 / cutlass::sizeof_bits<ElementD>::value,
            ElementAccumulator, ElementAccumulator>,
        // NOTE: Threadblock swizzling is currently not supported by
        // CUTLASS's grouped kernels. This parameter is passed in at present
        // to match the APIs of other kernels. The parameter is unused
        // within the kernel.
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 4>::GemmKernel;
    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    static void launchCombineScalesAndSetupAlphaPtrArray(float const* scale1, float const* scale2,
        float* combined_scale, float** alpha_ptr_array, int num_experts, cudaStream_t stream)
    {
        constexpr int BLOCK_SIZE = 256;
        dim3 block(BLOCK_SIZE);
        dim3 grid((num_experts + block.x - 1) / block.x);
        combimeScalesAndsetupAlphaPtrArrayKernel<<<grid, block, 0, stream>>>(
            scale1, scale2, combined_scale, alpha_ptr_array, num_experts);
    }

    static void call(GroupedGemmInput<T, WeightType, GemmOutputType, ScaleType, QuantModeType>& inputs)
    {
        // CPU_CHECK_FORMAT(false, "moeGemmTensorWise not impl");
        auto block_A = (ElementA*) inputs.A;
        auto block_B = (ElementB*) inputs.B;
        // auto block_C = (ElementC*)inputs.C;
        auto block_D = (ElementD*) inputs.D;

        auto ptr_A = (ElementA**) (inputs.workspace_ptr);
        auto ptr_B = (ElementB**) (ptr_A + inputs.num_experts);
        auto ptr_C = (ElementC**) (ptr_B + inputs.num_experts);
        auto ptr_D = (ElementD**) (ptr_C + inputs.num_experts);

        auto lda = (LeadingDimElement*) (align_pointer(ptr_D + inputs.num_experts));
        auto ldb = (LeadingDimElement*) (align_pointer(lda + inputs.num_experts));
        auto ldc = (LeadingDimElement*) (align_pointer(ldb + inputs.num_experts));
        auto ldd = (LeadingDimElement*) (align_pointer(ldc + inputs.num_experts));

        auto problem_size_workspace = (cutlass::gemm::GemmCoord*) (align_pointer(ldd + inputs.num_experts));

        auto combined_scale = (ScaleType*) (align_pointer(problem_size_workspace + inputs.num_experts));
        auto alpha_ptr_array = (ScaleType**) (align_pointer(combined_scale + inputs.num_experts));

        auto gemm_workspace = (uint8_t*) (align_pointer(alpha_ptr_array + inputs.num_experts));

        constexpr int BLOCK_SIZE = 128;
        auto GRID_SIZE = (inputs.num_experts + BLOCK_SIZE - 1) / BLOCK_SIZE;
        configureGemmKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE), 0, inputs.stream>>>(inputs.num_experts,
            inputs.cumsum_counter, block_A, block_B, block_D, inputs.N, inputs.K, ptr_A, ptr_B, ptr_D, lda, ldb, ldd,
            problem_size_workspace);

        launchCombineScalesAndSetupAlphaPtrArray(
            inputs.scales_a, inputs.scales_b, combined_scale, alpha_ptr_array, inputs.num_experts, inputs.stream);
        GemmGrouped gemm;
        typename GemmGrouped::EpilogueOutputOp::Params epilogue_op(alpha_ptr_array, nullptr);
        int threadblock_count = GemmGrouped::sufficient(nullptr, inputs.num_experts);

        typename GemmGrouped::Arguments arguments(problem_size_workspace, inputs.num_experts, threadblock_count,
            epilogue_op, ptr_A, ptr_B, ptr_C, ptr_D, lda, ldb, ldc, ldd, (cutlass::gemm::GemmCoord*) nullptr);
        // Check if the problem size is supported or not
        CUTLASS_CHECK(gemm.can_implement(arguments));

        // Initialize CUTLASS kernel with arguments and workspace pointer
        CUTLASS_CHECK(gemm.initialize(arguments, gemm_workspace, inputs.stream));

        // Correctness / Warmup iteration
        CUTLASS_CHECK(gemm.run(inputs.stream));
    }
};

constexpr static int DEQUANTIZE_THREADS_PER_BLOCK = 256;

__device__ inline int64_t findTotalEltsLessThanTarget(
    int const* d_cumsum_counter, int const num_experts, int const target)
{
    int expert_id = -1;
    for (int i = 0; i < num_experts; i++)
    {
        if (target >= d_cumsum_counter[i] && target < d_cumsum_counter[i + 1])
        {
            expert_id = i;
            break;
        }
    }
    return expert_id;
}

template <class InputType, class OutputType, class ScaleType>
__global__ void doDequantizePerchannel(InputType const* input, int const* d_cumsum_counter, int const num_experts,
    ScaleType const* act_scale, ScaleType const* weight_scale, OutputType* output, int64_t inter_size)
{
    // compute type , scale type 均保持为 float
    // step 1 . 找出每个 block 的对应的 expert
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;
    __shared__ int expert_id;
    if (tid == 0)
    {
        expert_id = findTotalEltsLessThanTarget(d_cumsum_counter, num_experts, token);
    }

    __syncthreads();

    // step 2 . dequantize
    InputType const* input_row = input + token * inter_size;
    ScaleType const* act_scale_row = act_scale + token;
    ScaleType const* weight_scale_row = weight_scale + expert_id * inter_size;
    OutputType* output_row = output + token * inter_size;

    constexpr int64_t DEQUANTIZE_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<InputType>::value;

    using DataElem = cutlass::Array<InputType, DEQUANTIZE_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, DEQUANTIZE_ELEM_PER_THREAD>;
    using OutputElem = cutlass::Array<OutputType, DEQUANTIZE_ELEM_PER_THREAD>;

    auto input_row_vec = reinterpret_cast<DataElem const*>(input_row);
    auto weight_scale_row_vec = reinterpret_cast<ComputeElem const*>(weight_scale_row);
    auto output_row_vec = reinterpret_cast<OutputElem*>(output_row);
    int64_t const start_offset = tid;
    int64_t const stride = DEQUANTIZE_THREADS_PER_BLOCK;
    assert(inter_size % DEQUANTIZE_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / DEQUANTIZE_ELEM_PER_THREAD;

    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto value = arrayConvert<DataElem, ComputeElem>(input_row_vec[elem_index]);
        output_row_vec[elem_index]
            = arrayConvert<ComputeElem, OutputElem>(value * weight_scale_row_vec[elem_index] * act_scale_row[0]);
    }
}

template <typename T, typename WeightType, typename GemmOutputType, typename ScaleType, typename arch,
    typename QuantModeType, typename std::enable_if_t<std::is_same_v<QuantModeType, QuantMode::FP8_ROWWISE>>* = nullptr>
struct moeGemmRowWise
{
    using PendingDequantType = float;
    using ElementA = typename NvToCutlassTypeAdapter<T>::type;          // Element type for
                                                                        // A matrix operand
    using ElementB = typename NvToCutlassTypeAdapter<WeightType>::type; // Element type for
                                                                        // B matrix operand
    using ElementC =
        typename NvToCutlassTypeAdapter<PendingDequantType>::type; // Element type for C and D matrix operands
    // D matrix configuration
    using ElementD = ElementC;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = LayoutC;

    constexpr static int ElementsPerAccessA = 128 / cutlass::sizeof_bits<ElementA>::value;
    constexpr static int ElementsPerAccessB = 128 / cutlass::sizeof_bits<ElementB>::value;
    using LeadingDimElement = int64_t;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<ElementA, LayoutA,
        cutlass::ComplexTransform::kNone, ElementsPerAccessA, ElementB, LayoutB, cutlass::ComplexTransform::kNone,
        ElementsPerAccessB, ElementD, LayoutD, ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
        cutlass::gemm::GemmShape<64, 128, 64>, cutlass::gemm::GemmShape<64, 32, 64>,
        cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombination<ElementD, 128 / cutlass::sizeof_bits<ElementD>::value, float,
            float, cutlass::epilogue::thread::ScaleType::Nothing>,
        // NOTE: Threadblock swizzling is currently not supported by CUTLASS's
        // grouped kernels. This parameter is passed in at present to match the
        // APIs of other kernels. The parameter is unused within the kernel.
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 4>::GemmKernel;
    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    static void launchDequantizePerchannel(PendingDequantType const* input, int const* d_cumsum_counter,
        int const num_experts, ScaleType const* act_scale, ScaleType const* weight_scale, GemmOutputType* output,
        int64_t num_elements, int64_t inter_size, cudaStream_t stream)
    {
        dim3 grid(num_elements);
        dim3 block(DEQUANTIZE_THREADS_PER_BLOCK);
        doDequantizePerchannel<PendingDequantType, GemmOutputType, ScaleType><<<grid, block, 0, stream>>>(
            input, d_cumsum_counter, num_experts, act_scale, weight_scale, output, inter_size);
        CUDA_CHECK(cudaGetLastError());
    }

    static void call(GroupedGemmInput<T, WeightType, GemmOutputType, ScaleType, QuantModeType>& inputs)
    {
        auto block_A = (ElementA*) inputs.A;
        auto block_B = (ElementB*) inputs.B;
        // auto block_C = (ElementC*)inputs.C;
        auto block_D = (ElementD*) inputs.pending_dq_output;

        auto ptr_A = (ElementA**) (inputs.workspace_ptr);
        auto ptr_B = (ElementB**) (ptr_A + inputs.num_experts);
        auto ptr_C = (ElementC**) (ptr_B + inputs.num_experts);
        auto ptr_D = (ElementD**) (ptr_C + inputs.num_experts);

        auto lda = (LeadingDimElement*) (align_pointer(ptr_D + inputs.num_experts));
        auto ldb = (LeadingDimElement*) (align_pointer(lda + inputs.num_experts));
        auto ldc = (LeadingDimElement*) (align_pointer(ldb + inputs.num_experts));
        auto ldd = (LeadingDimElement*) (align_pointer(ldc + inputs.num_experts));

        auto problem_size_workspace = (cutlass::gemm::GemmCoord*) (align_pointer(ldd + inputs.num_experts));

        auto gemm_workspace = (uint8_t*) (align_pointer(problem_size_workspace + inputs.num_experts));
        constexpr int BLOCK_SIZE = 128;
        auto GRID_SIZE = (inputs.num_experts + BLOCK_SIZE - 1) / BLOCK_SIZE;
        configureGemmKernel<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE), 0, inputs.stream>>>(inputs.num_experts,
            inputs.cumsum_counter, block_A, block_B, block_D, inputs.N, inputs.K, ptr_A, ptr_B, ptr_D, lda, ldb, ldd,
            problem_size_workspace);
        GemmGrouped gemm;
        typename GemmGrouped::EpilogueOutputOp::Params epilogue_op(1, 0);
        int threadblock_count = GemmGrouped::sufficient(nullptr, inputs.num_experts);
        typename GemmGrouped::Arguments arguments(problem_size_workspace, inputs.num_experts, threadblock_count,
            epilogue_op, ptr_A, ptr_B, ptr_C, ptr_D, lda, ldb, ldc, ldd, (cutlass::gemm::GemmCoord*) nullptr);
        // Check if the problem size is supported or not
        CUTLASS_CHECK(gemm.can_implement(arguments));

        // Initialize CUTLASS kernel with arguments and workspace pointer
        CUTLASS_CHECK(gemm.initialize(arguments, gemm_workspace, inputs.stream));
        CUTLASS_CHECK(gemm.run(inputs.stream));

        launchDequantizePerchannel(inputs.pending_dq_output, inputs.cumsum_counter, inputs.num_experts, inputs.scales_a,
            inputs.scales_b, inputs.D, inputs.num_rows * inputs.topk, inputs.N, inputs.stream);
    }
};

// A simple POD structure, can be constructed on host or device
// Corresponds to EVTD = Sm80EVT<D, Sm80EVT<Compute1, Sm80EVT<Compute0, Accum, ScaleA>, ScaleB>>
template <typename ScaleType, typename ElementD>
struct SimplifiedEpilogueParams
{
    // The memory layout must be exactly the same as EpilogueParams.
    //
    // Memory layout derivation (based on the recursive structure of Sm80EVT):
    //   EVTCompute0 = Sm80EVT<Compute0, Accum, ScaleA>
    //     -> Params = tuple<Accum::Params(empty), ScaleA::Params(16B), Compute0::Params(empty)>
    //     -> Total size: 32 bytes
    //
    //   EVTCompute1 = Sm80EVT<Compute1, EVTCompute0, ScaleB>
    //     -> Params = tuple<EVTCompute0::Params(32B), ScaleB::Params(16B), Compute1::Params(empty)>
    //     -> Total size: 56 bytes
    //
    //   EVTD = Sm80EVT<D, EVTCompute1>
    //     -> Params = tuple<EVTCompute1::Params(56B), D::Params(16B)>
    //     -> Total size: 72 bytes
    //
    // Actual memory layout (72 bytes total):
    //
    //   [EVTCompute1::Params - 56 bytes]
    //     [EVTCompute0::Params - 32 bytes]
    //       offset 0-7:   Accum::Params(empty,1B) + padding(7B) = 8 bytes
    //       offset 8-23:  ScaleA::Params (ptr_col 8B, null_default 4B, padding 4B) = 16 bytes
    //       offset 24-31: Compute0::Params(empty,1B) + padding(7B) = 8 bytes
    //     [ScaleB::Params - 16 bytes]
    //       offset 32-47: ScaleB::Params (ptr_row 8B, null_default 4B, padding 4B) = 16 bytes
    //     [Compute1::Params - 8 bytes]
    //       offset 48-55: Compute1::Params(empty,1B) + padding(7B) = 8 bytes
    //   [D::Params - 16 bytes]
    //     offset 56-71: D::Params (ptr_aux 8B, stride_m 8B) = 16 bytes
    //

    uint64_t accum_padding;       // offset 0:  Accum::Params(empty,1B) + padding(7B)
    ScaleType const* ptr_scale_a; // offset 8:  ScaleA::Params::ptr_col
    ScaleType null_default_a;     // offset 16: ScaleA::Params::null_default (4B)
    uint32_t padding1;            // offset 20: Padding after ScaleA stride (4B)
    uint64_t compute0_padding;    // offset 24: Compute0::Params(empty,1B) + padding(7B)
    ScaleType const* ptr_scale_b; // offset 32: ScaleB::Params::ptr_row
    ScaleType null_default_b;     // offset 40: ScaleB::Params::null_default (4B)
    uint32_t padding2;            // offset 44: Padding after ScaleB stride (4B)
    uint64_t compute1_padding;    // offset 48: Compute1::Params(empty,1B) + padding(7B)
    ElementD* ptr_d;              // offset 56: D::Params::ptr_aux
    int64_t stride_d_m;           // offset 64: D::Params::dAux (leading dimension)

    // Constructor
    CUTLASS_HOST_DEVICE
    SimplifiedEpilogueParams(ScaleType const* scale_a_ptr, ScaleType const* scale_b_ptr, ElementD* d_ptr,
        int64_t n // leading dimension of the output matrix
        )
        : accum_padding(0)
        , ptr_scale_a(scale_a_ptr)
        , null_default_a(0)
        , padding1(0)
        , compute0_padding(0)
        , ptr_scale_b(scale_b_ptr)
        , null_default_b(0)
        , padding2(0)
        , compute1_padding(0)
        , ptr_d(d_ptr)
        , stride_d_m(n)
    {
    }
};

template <typename ElementA, typename ElementB, typename ScaleType, typename ElementD, typename LeadingDimElement,
    typename ProblemShape, typename EpilogueParams>
__global__ void configureGemmKernelWithVisitor(int num_experts, int const* cumsum_counter, ElementA* A, ElementB* B,
    ScaleType const* scales_a, ScaleType const* scales_b, ElementD* D, int64_t N, int64_t K, ElementA** ptr_A,
    ElementB** ptr_B, LeadingDimElement* lda, LeadingDimElement* ldb, ProblemShape* problem_sizes_device,
    EpilogueParams* epilogue_params_device)
{
    static_assert(sizeof(SimplifiedEpilogueParams<ScaleType, ElementD>) == sizeof(EpilogueParams),
        "SimplifiedEpilogueParams size must match EpilogueParams size");

    int expert_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert_id < num_experts)
    {
        int start_idx = cumsum_counter[expert_id];
        int end_idx = cumsum_counter[expert_id + 1];
        int M = end_idx - start_idx;

        ptr_A[expert_id] = A + start_idx * K;
        ptr_B[expert_id] = B + expert_id * K * N;

        lda[expert_id] = K;
        // NOTE ColumnMajor
        ldb[expert_id] = K;

        ProblemShape problem = ProblemShape(M, N, K);
        problem_sizes_device[expert_id] = problem;

        ScaleType const* ptr_scale_A = scales_a + cumsum_counter[expert_id];
        ScaleType const* ptr_scale_B = scales_b + expert_id * N;
        ElementD* ptr_D = D + cumsum_counter[expert_id] * N;

        *reinterpret_cast<SimplifiedEpilogueParams<ScaleType, ElementD>*>(&epilogue_params_device[expert_id])
            = SimplifiedEpilogueParams<ScaleType, ElementD>(ptr_scale_A, ptr_scale_B, ptr_D, N);
    }
}

template <typename T, typename WeightType, typename GemmOutputType, typename ScaleType, typename arch,
    typename QuantModeType, typename std::enable_if_t<std::is_same_v<QuantModeType, QuantMode::FP8_ROWWISE>>* = nullptr>
struct moeGemmRowWiseFused
{
    using ElementA = typename NvToCutlassTypeAdapter<T>::type;          // Element type for
                                                                        // A matrix operand
    using ElementB = typename NvToCutlassTypeAdapter<WeightType>::type; // Element type for
                                                                        // B matrix operand
    using ElementC = GemmOutputType;
    // D matrix configuration
    using ElementD = ElementC;
    using ElementCompute = float;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = LayoutC;

    constexpr static int ElementsPerAccessA = 128 / cutlass::sizeof_bits<ElementA>::value;
    constexpr static int ElementsPerAccessB = 128 / cutlass::sizeof_bits<ElementB>::value;
    constexpr static int ElementsPerAccessC = 128 / cutlass::sizeof_bits<ElementC>::value;
    constexpr static int EVTEpilogueStages = 1;
    using LeadingDimElement = int64_t;

    using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 64>; // Threadblock-level tile size (concept: GemmShape)
    using WarpShape = cutlass::gemm::GemmShape<64, 32, 64>;         // Warp-level tile size (concept: GemmShape)
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;   // Instruction-level tile size (concept: GemmShape)

    using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<ThreadblockShape, WarpShape,
        ElementC, ElementsPerAccessC, EVTEpilogueStages>;

    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

    using ScaleA
        = cutlass::epilogue::threadblock::VisitorColBroadcast<OutputTileThreadMap, ScaleType, cute::Stride<_1, _0, _0>>;

    using ScaleB
        = cutlass::epilogue::threadblock::VisitorRowBroadcast<OutputTileThreadMap, ScaleType, cute::Stride<_0, _1, _0>>;

    using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTCompute0 = cutlass::epilogue::threadblock::Sm80EVT<Compute0, Accum, ScaleA>;

    using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTCompute1 = cutlass::epilogue::threadblock::Sm80EVT<Compute1, EVTCompute0, ScaleB>;

    using D = cutlass::epilogue::threadblock::VisitorAuxStore<OutputTileThreadMap, ElementD,
        cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, _1, _0>>;

    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<D, EVTCompute1>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGroupedWithVisitor<ElementA, LayoutA,
        cutlass::ComplexTransform::kNone, ElementsPerAccessA, ElementB, LayoutB, cutlass::ComplexTransform::kNone,
        ElementsPerAccessB, ElementD, LayoutD, ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm89,
        ThreadblockShape, WarpShape, InstructionShape, EVTD,
        // NOTE: Threadblock swizzling is currently not supported by CUTLASS's
        // grouped kernels. This parameter is passed in at present to match the
        // APIs of other kernels. The parameter is unused within the kernel.
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 4, EVTEpilogueStages>::GemmKernel;
    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    using FusionCallbacks = typename GemmKernel::FusionCallbacks;
    using EpilogueParams = typename FusionCallbacks::Params;

    static void call(GroupedGemmInput<T, WeightType, GemmOutputType, ScaleType, QuantModeType>& inputs)
    {
        auto block_A = (ElementA*) inputs.A;
        auto block_B = (ElementB*) inputs.B;
        // auto block_C = (ElementC*)inputs.C;
        auto block_D = (ElementD*) inputs.pending_dq_output;

        auto ptr_A = (ElementA**) (inputs.workspace_ptr);
        auto ptr_B = (ElementB**) (ptr_A + inputs.num_experts);

        auto lda = (LeadingDimElement*) (align_pointer(ptr_B + inputs.num_experts));
        auto ldb = (LeadingDimElement*) (align_pointer(lda + inputs.num_experts));

        auto problem_size_workspace = (cutlass::gemm::GemmCoord*) (align_pointer(ldb + inputs.num_experts));

        auto epilogue_params = (EpilogueParams*) (align_pointer(problem_size_workspace + inputs.num_experts));

        auto gemm_workspace = (uint8_t*) (align_pointer(epilogue_params + inputs.num_experts));

        constexpr int BLOCK_SIZE = 128;
        auto GRID_SIZE = (inputs.num_experts + BLOCK_SIZE - 1) / BLOCK_SIZE;
        configureGemmKernelWithVisitor<<<dim3(GRID_SIZE), dim3(BLOCK_SIZE), 0, inputs.stream>>>(inputs.num_experts,
            inputs.cumsum_counter, block_A, block_B, inputs.scales_a, inputs.scales_b, inputs.D, inputs.N, inputs.K,
            ptr_A, ptr_B, lda, ldb, problem_size_workspace, epilogue_params);

        int threadblock_count = GemmGrouped::sufficient(nullptr, inputs.num_experts);
        typename GemmGrouped::Arguments arguments(problem_size_workspace, inputs.num_experts, threadblock_count,
            epilogue_params, ptr_A, ptr_B, lda, ldb, (cutlass::gemm::GemmCoord*) nullptr);

        GemmGrouped gemm;

        // Check if the problem size is supported or not
        CUTLASS_CHECK(gemm.can_implement(arguments));

        // Initialize CUTLASS kernel with arguments and workspace pointer
        CUTLASS_CHECK(gemm.initialize(arguments, gemm_workspace, inputs.stream));
        CUTLASS_CHECK(gemm.run(inputs.stream));
    }
};

template <typename T,          /*The type used for activations/scales/compute*/
    typename WeightType,       /* The type for the MoE weights */
    typename OutputType,       /* The output type for the GEMM */
    typename ScaleType = float /* The type for the scales/bias */
    >
class MoeGemmRunner
{
public:
    MoeGemmRunner();

    static constexpr bool use_fp8 = (std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>);

    template <typename QuantModeType>
    void dispatchToArch(GroupedGemmInput<T, WeightType, OutputType, ScaleType, QuantModeType> inputs);

    template <typename QuantModeType>
    void runGemm(GroupedGemmInput<T, WeightType, OutputType, ScaleType, QuantModeType> inputs);
    size_t getMaxWorkspaceSize(int num_experts, QuantMode quant_mode) const;

    int getSM() const;

private:
    int sm_{};
    int multi_processor_count_{};
    // cutlass::KernelHardwareInfo kernel_hw_info_;
};

template <typename T, typename WeightType, typename OutputType, typename ScaleType>
MoeGemmRunner<T, WeightType, OutputType, ScaleType>::MoeGemmRunner()
{
    int device{-1};
    cudaGetDevice(&device);
    int sm_major = 0;
    int sm_minor = 0;
    cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device);
    sm_ = sm_major * 10 + sm_minor;
    //   std::cout << "Current SM version: " << sm_ << std::endl;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleType>
int MoeGemmRunner<T, WeightType, OutputType, ScaleType>::getSM() const
{
    return this->sm_;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleType>
size_t MoeGemmRunner<T, WeightType, OutputType, ScaleType>::getMaxWorkspaceSize(
    int num_experts, QuantMode quant_mode) const
{
    // 预分配 CUTLASS GEMM 所需参数空间
    // 包含：stride、ptr、problem size、leading dimension 和 workspace
    // 经验表明 18MB 可满足当前所有 GEMM 场景，避免重复计算开销
    return 18 * 1024 * 1024;
}

template <typename T, typename WeightType, typename OutputType, typename ScaleType>
template <typename QuantModeType>
void MoeGemmRunner<T, WeightType, OutputType, ScaleType>::runGemm(
    GroupedGemmInput<T, WeightType, OutputType, ScaleType, QuantModeType> inputs)
{
    dispatchToArch<QuantModeType>(inputs);
}

template <typename T, typename WeightType, typename GemmOutputType, typename ScaleType, typename arch,
    typename QuantModeType>
void dispatchMoeGemm(GroupedGemmInput<T, WeightType, GemmOutputType, ScaleType, QuantModeType> inputs)
{
    if constexpr (arch::kMinComputeCapability >= 89 && std::is_same_v<QuantModeType, QuantMode::FP8_ROWWISE>)
    {
        if (inputs.fused)
        {
            moeGemmRowWiseFused<T, WeightType, GemmOutputType, ScaleType, arch, QuantModeType>::call(inputs);
        }
        else
        {
            moeGemmRowWise<T, WeightType, GemmOutputType, ScaleType, arch, QuantModeType>::call(inputs);
        }
    }
    else if constexpr (arch::kMinComputeCapability >= 89 && std::is_same_v<QuantModeType, QuantMode::FP8_QDQ>)
    {
        moeGemmTensorWise<T, WeightType, GemmOutputType, ScaleType, arch, QuantModeType>::call(inputs);
    }
    else if constexpr (arch::kMinComputeCapability >= 80 && std::is_same_v<QuantModeType, QuantMode::NONE>)
    {
        moeGemm<T, WeightType, GemmOutputType, ScaleType, arch, QuantModeType>::call(inputs);
    }
    else
    {
        CPU_CHECK_FORMAT(false, "Quant mode %s not impl in sm %d. ", inputs.quant_mode.toQuantAlgo(), inputs.sm);
    }
}

template <typename T, typename WeightType, typename GemmOutputType, typename ScaleType, typename arch,
    typename QuantModeType>
void dispatchMoeGemmTmaWarpSpecialized(GroupedGemmInput<T, WeightType, GemmOutputType, ScaleType, QuantModeType> inputs)
{
    if constexpr (std::is_same_v<QuantModeType, QuantMode::FP8_BLOCKWISE>)
    {
        if (inputs.K <= 128)
        {
            moeGemmTmaWarpSpecializedBlockWise<T, WeightType, GemmOutputType, ScaleType, arch, QuantModeType, CooperativeConfig>::call(inputs);
        } else
        {
            moeGemmTmaWarpSpecializedBlockWise<T, WeightType, GemmOutputType, ScaleType, arch, QuantModeType, PingpongConfig>::call(inputs);
        }
    }
    else if constexpr (std::is_same_v<QuantModeType, QuantMode::FP8_ROWWISE>
        || std::is_same_v<QuantModeType, QuantMode::FP8_QDQ> || std::is_same_v<T, float>)
    {
        // std::cout << "[警告]: " << inputs.operation_name
        //           << " 使用的 quant mode: " << inputs.quant_mode.toQuantAlgo()
        //           << " 目前没有 TMA warp specialized 的 kernel 实现, "
        //              "将自动降级到普通实现, "
        //              "这是为保证向后兼容旧硬件的低效算法实现, 不推荐在 sm "
        //           << inputs.sm << " 上使用. " << std::endl;
        dispatchMoeGemm<T, WeightType, GemmOutputType, ScaleType, arch, QuantModeType>(inputs);
    }
    else
    {
        moeGemmTmaWarpSpecialized<T, WeightType, GemmOutputType, ScaleType, arch, QuantModeType>::call(inputs);
    }
}

template <typename T, typename WeightType, typename OutputType, typename ScaleType>
template <typename QuantModeType>
void MoeGemmRunner<T, WeightType, OutputType, ScaleType>::dispatchToArch(
    GroupedGemmInput<T, WeightType, OutputType, ScaleType, QuantModeType> inputs)
{
    static_assert(use_fp8
            == (std::is_same_v<QuantModeType, QuantMode::FP8_QDQ>
                || std::is_same_v<QuantModeType, QuantMode::FP8_BLOCKWISE>
                || std::is_same_v<QuantModeType, QuantMode::FP8_ROWWISE>),
        "Class instantiation mismatch detected between `use_fp8` compile-time "
        "flag and `QuantModeType`.\n"
        "  When `use_fp8` is true, `QuantModeType` must be one of: "
        "QuantMode::FP8_QDQ, QuantMode::FP8_BLOCKWISE, "
        "QuantMode::FP8_ROWWISE.\n"
        "  When `use_fp8` is false, `QuantModeType` must NOT be an FP8 mode.\n"
        "Possible fix: Ensure the `QuantModeType` template argument aligns "
        "with the `use_fp8` setting in class "
        "instantiation.");
    if (sm_ >= 80 && sm_ < 90)
    {
        if constexpr (use_fp8)
        {
            CPU_CHECK_FORMAT(sm_ == 89,
                "For sm >= 80 and < 90, fp8 is only supported "
                "with sm == 89, current sm is %d",
                sm_);
            dispatchMoeGemm<T, WeightType, OutputType, ScaleType, cutlass::arch::Sm89, QuantModeType>(inputs);
        }
        else
        {
            dispatchMoeGemm<T, WeightType, OutputType, ScaleType, cutlass::arch::Sm80, QuantModeType>(inputs);
        }
    }

    else if (sm_ == 90)
    {
        dispatchMoeGemmTmaWarpSpecialized<T, WeightType, OutputType, ScaleType, cutlass::arch::Sm90, QuantModeType>(
            inputs);
    }
}