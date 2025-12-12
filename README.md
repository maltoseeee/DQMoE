# Dynamic Quantization MoE (DQMoE)

一个基于 CUDA 和 CUTLASS 的高性能 Dynamic Quantization Mixture of Experts (MoE) 推理库。

## RoadMap
- [x] 支持 FP8 量化 MoE
- [x] 支持 BF16, FP16, FP32 实现
- [x] 支持 Tensor-wise, Channel-wise 量化
- [x] 支持 1 X 128 X 128 Block-wise 量化
- [x] 提供性能更好 `ada` FP8 GEMM fused kernel
- [ ] 运用高性能的 `DeepGEMM` 实现

## 主要特性

- 🚀 基于 CUTLASS 的高性能 CUDA 内核
- 📊 支持多种数据类型：BF16, FP16, FP8
- 🔧 支持 Tensor-wise, Channel-wise 和 1 X 128 X 128 Block-wise 量化
- 🧪 完整的测试和性能基准测试

## 系统要求

- CUDA 12.4+
- PyTorch 2.1+
- Python 3.8+
- NVIDIA GPU (Compute Capability 8.9+)

## 安装

### 1. 克隆项目
```bash
git clone --recursive https://github.com/maltoseeee/moe.git
cd moe
```

### 2. 编译安装
```bash
# 开发模式编译
bash develop.sh

# 测试所有的实现
cd test && python torch_test.py
```

## 许可证

本项目遵循开源许可证，具体请查看项目中的许可证文件。

## 致谢

DQMoE 的设计灵感来源于 CUTLASS, DeepGEMM, TRTLLM，感谢并致敬各位开发者！


