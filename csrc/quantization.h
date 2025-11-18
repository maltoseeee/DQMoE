#pragma once
#include <optional>
#include <string>

class QuantMode
{
public:
    using BaseType = std::uint32_t;

    struct NONE
    {
        static constexpr BaseType value = 0;
    };

    struct FP8_QDQ
    {
        static constexpr BaseType value = 1u << 8;
    };

    struct FP8_ROWWISE
    {
        static constexpr BaseType value = 1u << 9;
    };

    struct FP8_BLOCKWISE
    {
        static constexpr BaseType value = 1u << 10;
    };

    struct FP4
    {
        static constexpr BaseType value = 1u << 12;
    };

    struct W4A8_MXFP4_FP8
    {
        static constexpr BaseType value = 1u << 14;
    };

    explicit constexpr QuantMode(BaseType value) noexcept
        : mValue{value}
    {
    }

    QuantMode() noexcept = default;

    constexpr QuantMode(QuantMode const&) noexcept = default;

    constexpr QuantMode& operator=(QuantMode const& other) noexcept = default;

    static constexpr QuantMode none() noexcept
    {
        return QuantMode(BaseType(0));
    }

    static constexpr QuantMode int4Weights() noexcept
    {
        return QuantMode(BaseType(1u) << 0);
    }

    static constexpr QuantMode int8Weights() noexcept
    {
        return QuantMode(BaseType(1u) << 1);
    }

    static constexpr QuantMode fp8Qdq() noexcept
    {
        return QuantMode(BaseType(1u) << 8);
    }

    static constexpr QuantMode fp8RowWise() noexcept
    {
        return QuantMode(BaseType(1u) << 9);
    }

    static constexpr QuantMode fp8BlockWise() noexcept
    {
        return QuantMode(BaseType(1u) << 10);
    }

    static constexpr QuantMode nvfp4() noexcept
    {
        return QuantMode(BaseType(1u) << 12);
    }

    static constexpr QuantMode w4a8Mxfp4Fp8() noexcept
    {
        return QuantMode(BaseType(1u) << 14);
    }

    constexpr BaseType value() const noexcept
    {
        return mValue;
    }

    constexpr bool isSet(QuantMode const& mode) const noexcept
    {
        return (mValue & mode.value()) == mode.value();
    }

    constexpr bool hasFp8Qdq() const noexcept
    {
        return isSet(fp8Qdq());
    }

    constexpr bool hasFp8RowWise() const noexcept
    {
        return isSet(fp8RowWise());
    }

    constexpr bool hasFp8BlockWise() const noexcept
    {
        return isSet(fp8BlockWise());
    }

    constexpr bool hasNvfp4() const noexcept
    {
        return isSet(nvfp4());
    }

    constexpr bool hasW4a8Mxfp4Fp8() const noexcept
    {
        return isSet(w4a8Mxfp4Fp8());
    }

    static constexpr QuantMode fromDescription(
        bool useFp8Qdq, bool useFp8RowWise, bool useFp4Quant, bool usefp8BlockWise, bool useW4a8Mxfp4Fp8)
    {
        QuantMode quantMode{};

        if (useFp8Qdq)
        {
            quantMode += fp8Qdq();
        }

        if (useFp8RowWise)
        {
            quantMode += fp8RowWise();
        }

        if (usefp8BlockWise)
        {
            quantMode += fp8BlockWise();
        }

        if (useFp4Quant)
        {
            quantMode += nvfp4();
        }

        if (useW4a8Mxfp4Fp8)
        {
            quantMode += w4a8Mxfp4Fp8();
        }

        return quantMode;
    }

    static QuantMode const fromQuantAlgo(std::optional<std::string> quantAlgo = std::nullopt)
    {
        QuantMode quantMode{};
        if (quantAlgo == "FP8")
        {
            quantMode = fromDescription(true, false, false, false, false);
        }
        else if (quantAlgo == "FP8_ROWWISE")
        {
            quantMode = fromDescription(false, true, true, false, false);
        }
        else if (quantAlgo == "FP4")
        {
            quantMode = fromDescription(false, false, true, false, false);
        }
        else if (quantAlgo == "FP8_BLOCKWISE")
        {
            quantMode = fromDescription(false, false, false, true, false);
        }
        else if (quantAlgo == "W4A8_MXFP4_FP8")
        {
            quantMode = fromDescription(false, false, false, false, true);
        }

        return quantMode;
    }

    std::string const toQuantAlgo()
    {
        if (isSet(fp8Qdq()))
        {
            return "FP8";
        }
        else if (isSet(fp8RowWise()))
        {
            return "FP8_ROWWISE";
        }
        else if (isSet(nvfp4()))
        {
            return "FP4";
        }
        else if (isSet(fp8BlockWise()))
        {
            return "FP8_BLOCKWISE";
        }
        else if (isSet(w4a8Mxfp4Fp8()))
        {
            return "W4A8_MXFP4_FP8";
        }
        else
        {
            return "NONE";
        }
    }

    constexpr QuantMode operator+(QuantMode const& other) const noexcept
    {
        return QuantMode(mValue | other.mValue);
    }

    constexpr QuantMode& operator+=(QuantMode const& other) noexcept
    {
        return *this = *this + other;
    }

    constexpr QuantMode operator-(QuantMode const& other) const noexcept
    {
        return QuantMode(mValue & ~other.mValue);
    }

    constexpr QuantMode& operator-=(QuantMode const& other) noexcept
    {
        return *this = *this - other;
    }

    constexpr bool operator==(QuantMode const& other) const noexcept
    {
        return mValue == other.mValue;
    }

    constexpr bool operator!=(QuantMode const& other) const noexcept
    {
        return !(*this == other);
    }

private:
    BaseType mValue;
};

struct QuantParams
{
    // Int weight only quantization params
    struct
    {
        void const* fc1_weight_scales = nullptr;
        void const* fc2_weight_scales = nullptr;
    } wo;

    // FP8 quantization params
    struct
    {
        float const* act_scales = nullptr;
        float const* weight1_scales = nullptr;
        float const* weight2_scales = nullptr;
    } fp8;

    // GPTQ/AWQ quantization params
    struct GroupwiseInputs
    {
        struct GroupwiseGemmInputs
        {
            void const* act_scales = nullptr;
            void const* weight_scales = nullptr;
            void const* weight_zeros = nullptr;
            float const* alpha = nullptr;
        };

        int group_size = -1;
        GroupwiseGemmInputs fc1;
        GroupwiseGemmInputs fc2;
    } groupwise;

    // FP8 blockscaling params (for Deepseek)
    struct BlockScaleParams
    {
        float const* fc1_scales_ptrs = nullptr;
        float const* fc2_scales_ptrs = nullptr;

        BlockScaleParams() = default;

        BlockScaleParams(float const* fc1_scales_ptrs, float const* fc2_scales_ptrs)
            : fc1_scales_ptrs(fc1_scales_ptrs)
            , fc2_scales_ptrs(fc2_scales_ptrs)
        {
        }
    } fp8_block_scaling;

    static QuantParams Int(void const* fc1_weight_scales, void const* fc2_weight_scales)
    {
        QuantParams qp;
        qp.wo = {fc1_weight_scales, fc2_weight_scales};
        return qp;
    }

    static QuantParams FP8(float const* act_scales, float const* weight1_scales, float const* weight2_scales)
    {
        QuantParams qp;
        qp.fp8 = {act_scales, weight1_scales, weight2_scales};
        return qp;
    }

    static QuantParams GroupWise(int group_size, void const* fc1_weight_scales, void const* fc2_weight_scales,
        void const* fc1_activation_scales = nullptr, void const* fc2_activation_scales = nullptr,
        void const* fc1_weight_zeros = nullptr, void const* fc2_weight_zeros = nullptr,
        float const* fc1_alpha = nullptr, float const* fc2_alpha = nullptr)
    {
        QuantParams qp;
        qp.groupwise.group_size = group_size;
        qp.groupwise.fc1 = {fc1_activation_scales, fc1_weight_scales, fc1_weight_zeros, fc1_alpha};
        qp.groupwise.fc2 = {fc2_activation_scales, fc2_weight_scales, fc2_weight_zeros, fc2_alpha};
        return qp;
    }

    static QuantParams FP8BlockScaling(float const* fc1_scales, float const* fc2_scales)
    {
        QuantParams qp;
        qp.fp8_block_scaling = {fc1_scales, fc2_scales};
        return qp;
    }
};