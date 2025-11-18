#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
#include <string>

template <typename T>
struct NvToCutlassTypeAdapter
{
    using type = T;
};

template <>
struct NvToCutlassTypeAdapter<half>
{
    using type = cutlass::half_t;
};

template <>
struct NvToCutlassTypeAdapter<__nv_bfloat16>
{
    using type = cutlass::bfloat16_t;
};

template <>
struct NvToCutlassTypeAdapter<__nv_fp8_e4m3>
{
    using type = cutlass::float_e4m3_t;
};

template <class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const& input)
{
    using Type = typename U::Element;
    static_assert(T::kElements == U::kElements);
    U u;
#pragma unroll
    for (int i = 0; i < U::kElements; i++)
    {
        u[i] = static_cast<Type>(input[i]);
    }
    return u;
}

// Constants
static constexpr int FP8_BLOCK_SIZE = 256;
static constexpr size_t CUDA_MEM_ALIGN = 256;

static constexpr int SFVecSizeM = 1;
static constexpr int SFVecSizeN = 128;
static constexpr int SFVecSizeK = 128;

#define CUTLASS_CHECK(status)                                                                                          \
    {                                                                                                                  \
        cutlass::Status error = status;                                                                                \
        if (error != cutlass::Status::kSuccess)                                                                        \
        {                                                                                                              \
            std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ << std::endl;   \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    }

#define CUDA_CHECK(status)                                                                                             \
    {                                                                                                                  \
        cudaError_t error = status;                                                                                    \
        if (error != cudaSuccess)                                                                                      \
        {                                                                                                              \
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) << " at line: " << __LINE__              \
                      << std::endl;                                                                                    \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    }

// Alignment verification function for debugging
static inline bool is_aligned(void const* ptr, size_t align_size = CUDA_MEM_ALIGN)
{
    return (reinterpret_cast<size_t>(ptr) % align_size) == 0;
}

static inline void verify_alignment(void const* ptr, char const* name, size_t align_size = CUDA_MEM_ALIGN)
{
    if (!is_aligned(ptr, align_size))
    {
        std::cerr << "WARNING: " << name << " is not " << align_size << "-byte aligned! "
                  << "Address: " << ptr << ", Offset: " << (reinterpret_cast<size_t>(ptr) % align_size) << std::endl;
    }
}

// Memory bounds verification function for debugging
static inline void verify_memory_bounds(
    void const* ptr, size_t size, void const* workspace_start, size_t workspace_size, char const* name)
{
    char const* ptr_char = static_cast<char const*>(ptr);
    char const* workspace_start_char = static_cast<char const*>(workspace_start);
    char const* workspace_end_char = workspace_start_char + workspace_size;
    char const* ptr_end_char = ptr_char + size;

    if (ptr_char < workspace_start_char || ptr_end_char > workspace_end_char)
    {
        std::cerr << "ERROR: " << name << " memory bounds violation!" << std::endl;
        std::cerr << "  Workspace range: [" << workspace_start_char << ", " << workspace_end_char << ")" << std::endl;
        std::cerr << "  " << name << " range: [" << ptr_char << ", " << ptr_end_char << ")" << std::endl;
        std::cerr << "  Size: " << size << " bytes" << std::endl;

        if (ptr_char < workspace_start_char)
        {
            std::cerr << "  Underflow by: " << (workspace_start_char - ptr_char) << " bytes" << std::endl;
        }
        if (ptr_end_char > workspace_end_char)
        {
            std::cerr << "  Overflow by: " << (ptr_end_char - workspace_end_char) << " bytes" << std::endl;
        }
    }
}

// Memory alignment utility functions
static inline void* align_pointer(void* ptr, size_t align_size = CUDA_MEM_ALIGN)
{
    size_t uptr = reinterpret_cast<size_t>(ptr);
    if (uptr % align_size)
    {
        uptr += align_size - uptr % align_size;
    }
    return reinterpret_cast<void*>(uptr);
}

static inline size_t align_size(size_t size, size_t align_size)
{
    return (size + align_size - 1) & ~(align_size - 1);
}

namespace detail
{

inline void throwCheckError(char const* file, int line, char const* condition, char const* message = nullptr)
{
    std::ostringstream oss;
    oss << "[ASSERT FAILED] " << file << ":" << line << "  Condition: " << condition;
    if (message && message[0] != '\0')
    {
        oss << "  Info: " << message;
    }
    throw std::runtime_error(oss.str());
}

template <typename... Args>
std::string formatMessage(char const* fmt, Args&&... args)
{
    // 计算需要的缓冲区大小
    int size = snprintf(nullptr, 0, fmt, args...);
    if (size <= 0)
        return "";

    // 分配缓冲区并格式化
    std::vector<char> buf(size + 1);
    snprintf(buf.data(), buf.size(), fmt, args...);
    return std::string(buf.data());
}

} // namespace detail

#define LIKELY(x) __builtin_expect((x), 1)
#define UNLIKELY(x) __builtin_expect((x), 0)

#define CPU_CHECK(val)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        LIKELY(static_cast<bool>(val))                                                                                 \
        ? ((void) 0) : ::detail::throwCheckError(__FILE__, __LINE__, #val);                                            \
    } while (0)

#define CPU_CHECK_WITH_INFO(val, info)                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        LIKELY(static_cast<bool>(val))                                                                                 \
        ? ((void) 0) : ::detail::throwCheckError(__FILE__, __LINE__, #val, info);                                      \
    } while (0)

#define CPU_CHECK_FORMAT(val, fmt, ...)                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if (UNLIKELY(!static_cast<bool>(val)))                                                                         \
        {                                                                                                              \
            std::string message = ::detail::formatMessage(fmt, ##__VA_ARGS__);                                         \
            ::detail::throwCheckError(__FILE__, __LINE__, #val, message.c_str());                                      \
        }                                                                                                              \
    } while (0)
