import tensorrt as trt
import numpy as np
import ctypes
import torch
import os

# Load plugin
ctypes.CDLL("./libtrt_plugin.so", mode=ctypes.RTLD_GLOBAL | 0x0001)

# Parameters (global)
max_num_tokens = 2048
num_experts = 64
top_k = 3
hidden_size = 128
inter_size = 128

# === TensorRT dtype ↔ PyTorch dtype ===
_trt_to_torch_dtype_dict = {
    trt.float16: torch.float16,
    trt.float32: torch.float32,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
    trt.fp8: getattr(
        torch, "float8_e4m3fn", torch.uint8
    ),  # fallback to uint8 if FP8 not available
}


def trt_dtype_to_torch(dtype):
    ret = _trt_to_torch_dtype_dict.get(dtype)
    if ret is None:
        raise ValueError(f"Unsupported TRT dtype: {dtype}")
    return ret


# === Build with cache & skip if exists ===
def build_engine_if_not_exists(engine_path, build_func):
    if os.path.exists(engine_path):
        print(f"✅ Engine already exists: {engine_path}")
        return
    print(f"🔧 Building engine: {engine_path}")
    build_func()


# === FP8 Engine Builder (with Build Cache) ===
def build_fp8_engine():
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)  # Use INFO to see cache hits
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )

    fp8_max_val = 448.0
    eps = np.finfo(np.float32).eps

    # Inputs: already in ColumnMajor layout
    input_tensor = network.add_input(
        "input", trt.DataType.FLOAT, (max_num_tokens, hidden_size)
    )
    weight1_cm = network.add_input(
        "weight1", trt.DataType.FLOAT, (num_experts, inter_size * 2, hidden_size)
    )
    weight2_cm = network.add_input(
        "weight2", trt.DataType.FLOAT, (num_experts, hidden_size, inter_size)
    )
    index_tensor = network.add_input(
        "index", trt.DataType.INT32, (max_num_tokens, top_k)
    )

    def broadcast_help(network, ref_tensor, scalar):
        shape = [1] * len(ref_tensor.shape)
        layer = network.add_shuffle(scalar)
        layer.reshape_dims = tuple(shape)
        return layer.get_output(0)

    dummy_scale = network.add_constant(
        (1,), np.array([1.0], dtype=np.float32)
    ).get_output(0)
    fp8_max_const = network.add_constant(
        (1,), np.array([fp8_max_val], dtype=np.float32)
    ).get_output(0)
    eps_const = network.add_constant(
        (1,), np.array([eps], dtype=np.float32)
    ).get_output(0)

    def quantize_to_fp8(network, tensor, axis, fp8_max, eps):
        abs_tensor = network.add_unary(tensor, trt.UnaryOperation.ABS).get_output(0)
        reduce_max = network.add_reduce(
            abs_tensor, trt.ReduceOperation.MAX, axes=1 << axis, keep_dims=True
        ).get_output(0)
        fp8_max_b = broadcast_help(network, reduce_max, fp8_max)
        scale_raw = network.add_elementwise(
            reduce_max, fp8_max_b, trt.ElementWiseOperation.DIV
        ).get_output(0)
        eps_b = broadcast_help(network, scale_raw, eps)
        scale = network.add_elementwise(
            scale_raw, eps_b, trt.ElementWiseOperation.MAX
        ).get_output(0)
        quantized = network.add_elementwise(
            tensor, scale, trt.ElementWiseOperation.DIV
        ).get_output(0)
        quantized = network.add_quantize(
            quantized, dummy_scale, trt.DataType.FP8
        ).get_output(0)
        return quantized, scale

    weight1_fp8, w1_scale = quantize_to_fp8(
        network, weight1_cm, axis=2, fp8_max=fp8_max_const, eps=eps_const
    )
    weight2_fp8, w2_scale = quantize_to_fp8(
        network, weight2_cm, axis=2, fp8_max=fp8_max_const, eps=eps_const
    )
    input_fp8, actscale = quantize_to_fp8(
        network, input_tensor, axis=1, fp8_max=fp8_max_const, eps=eps_const
    )

    input_fp8.name = "input_fp8"
    weight1_fp8.name = "weight1_fp8"
    weight2_fp8.name = "weight2_fp8"
    w1_scale.name = "w1_scale"
    w2_scale.name = "w2_scale"
    actscale.name = "actscale"
    network.mark_output(input_fp8)
    network.mark_output(weight1_fp8)
    network.mark_output(weight2_fp8)
    network.mark_output(w1_scale)
    network.mark_output(w2_scale)
    network.mark_output(actscale)

    # Dummy biases (use_bias=0)
    bias1 = network.add_constant(
        (num_experts, inter_size * 2, hidden_size),
        np.zeros((num_experts, inter_size * 2, hidden_size), dtype=np.float32),
    ).get_output(0)
    bias2 = network.add_constant(
        (num_experts, hidden_size, inter_size),
        np.zeros((num_experts, hidden_size, inter_size), dtype=np.float32),
    ).get_output(0)

    # Plugin
    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator("mixture_of_experts", "1.0")
    assert creator, "Plugin not found!"

    fields = [
        trt.PluginField(
            "number_of_experts",
            np.array([num_experts], np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "top_k", np.array([top_k], np.int32), trt.PluginFieldType.INT32
        ),
        trt.PluginField(
            "expert_hidden_size",
            np.array([hidden_size], np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "expert_inter_size",
            np.array([inter_size], np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "max_num_tokens",
            np.array([max_num_tokens], np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField("use_bias", np.array([0], np.int32), trt.PluginFieldType.INT32),
        trt.PluginField(
            "type",
            np.array([int(trt.DataType.FP8)], np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "output_type",
            np.array([int(trt.DataType.HALF)], np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "quant_mode", np.array([1 << 9], np.int32), trt.PluginFieldType.INT32
        ),
    ]
    plugin = creator.create_plugin(
        "mixture_of_experts", trt.PluginFieldCollection(fields)
    )

    inputs = [
        input_fp8,
        weight1_fp8,
        weight2_fp8,
        index_tensor,
        bias1,
        bias2,
        actscale,
        w1_scale,
        w2_scale,
    ]
    layer = network.add_plugin_v2(inputs, plugin)

    layer = network.add_cast(layer.get_output(0), trt.DataType.FLOAT)
    layer.get_output(0).name = "output"
    network.mark_output(layer.get_output(0))

    engine = builder.build_serialized_network(network, config)
    assert engine, "FP8 engine build failed!"
    if isinstance(engine, trt.ICudaEngine):
        engine = engine.serialize()
    with open("test_fp8.engine", "wb") as f:
        f.write(engine)


# === FP32 Engine Builder ===
def build_fp32_engine():
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    config = builder.create_builder_config()

    input_tensor = network.add_input(
        "input", trt.DataType.FLOAT, (max_num_tokens, hidden_size)
    )
    weight1 = network.add_input(
        "weight1", trt.DataType.FLOAT, (num_experts, hidden_size, inter_size * 2)
    )
    weight2 = network.add_input(
        "weight2", trt.DataType.FLOAT, (num_experts, inter_size, hidden_size)
    )
    index_tensor = network.add_input(
        "index", trt.DataType.INT32, (max_num_tokens, top_k)
    )

    bias1 = network.add_constant(
        weight1.shape, np.zeros(tuple(weight1.shape), dtype=np.float32)
    ).get_output(0)
    bias2 = network.add_constant(
        weight2.shape, np.zeros(tuple(weight2.shape), dtype=np.float32)
    ).get_output(0)

    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator("mixture_of_experts", "1.0")
    assert creator, "Plugin not found!"

    fields = [
        trt.PluginField(
            "number_of_experts",
            np.array([num_experts], np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "top_k", np.array([top_k], np.int32), trt.PluginFieldType.INT32
        ),
        trt.PluginField(
            "expert_hidden_size",
            np.array([hidden_size], np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "expert_inter_size",
            np.array([inter_size], np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "max_num_tokens",
            np.array([max_num_tokens], np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField("use_bias", np.array([0], np.int32), trt.PluginFieldType.INT32),
        trt.PluginField(
            "type",
            np.array([int(trt.DataType.FLOAT)], np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "output_type",
            np.array([int(trt.DataType.FLOAT)], np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "quant_mode", np.array([0], np.int32), trt.PluginFieldType.INT32
        ),
    ]
    plugin = creator.create_plugin(
        "mixture_of_experts", trt.PluginFieldCollection(fields)
    )

    inputs = [input_tensor, weight1, weight2, index_tensor, bias1, bias2]
    layer = network.add_plugin_v2(inputs, plugin)
    layer.get_output(0).name = "output"
    network.mark_output(layer.get_output(0))

    engine = builder.build_serialized_network(network, config)
    assert engine, "FP32 engine build failed!"
    if isinstance(engine, trt.ICudaEngine):
        engine = engine.serialize()
    with open("test_fp32.engine", "wb") as f:
        f.write(engine)


# === Inference ===
def infer_trt_torch(engine_path, buffer_dict):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    stream = torch.cuda.Stream()

    # Set all tensor addresses
    for name in engine:
        if name in buffer_dict:
            assert buffer_dict[name].is_contiguous(), f"{name} must be contiguous"
            context.set_tensor_address(name, buffer_dict[name].data_ptr())
        else:
            # Allocate output
            dtype = trt_dtype_to_torch(engine.get_tensor_dtype(name))
            shape = context.get_tensor_shape(name)
            buffer_dict[name] = torch.empty(tuple(shape), dtype=dtype, device="cuda")
            context.set_tensor_address(name, buffer_dict[name].data_ptr())

    context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    return buffer_dict


# === Per-channel Quantization (for reference) ===
def per_channel_quantize(x, quant_axis, dtype="fp8", eps=1e-8):
    assert dtype == "fp8"
    abs_max = x.abs().amax(dim=quant_axis, keepdim=True)
    scale = abs_max.clamp(min=eps) / 448.0
    x_scaled = x / scale
    x_clamped = torch.clamp(x_scaled, -448.0, 448.0)
    return x_clamped, scale


# === Main ===
if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = "cuda"

    # Build engines only if not exists
    build_engine_if_not_exists("test_fp32.engine", build_fp32_engine)
    build_engine_if_not_exists("test_fp8.engine", build_fp8_engine)

    # Generate data
    std = 0.05
    input_t = torch.normal(0, std, (max_num_tokens, hidden_size), device=device)
    weight1_t = torch.normal(
        0, std, (num_experts, hidden_size, inter_size * 2), device=device
    )
    weight2_t = torch.normal(
        0, std, (num_experts, inter_size, hidden_size), device=device
    )
    index_t = torch.randint(
        0, num_experts, (max_num_tokens, top_k), device=device, dtype=torch.int32
    )

    # FP32 uses original layout
    input_dict_fp32 = {
        "input": input_t,
        "weight1": weight1_t,
        "weight2": weight2_t,
        "index": index_t,
    }

    # FP8 uses transposed layout (ColumnMajor)
    input_dict_fp8 = {
        "input": input_t,
        "weight1": weight1_t.permute(0, 2, 1).contiguous(),
        "weight2": weight2_t.permute(0, 2, 1).contiguous(),
        "index": index_t,
    }

    # Inference
    out_fp32 = infer_trt_torch("test_fp32.engine", input_dict_fp32)["output"]
    out_fp8_dict = infer_trt_torch("test_fp8.engine", input_dict_fp8)
    out_fp8 = out_fp8_dict["output"]
    print(out_fp32)
    print(out_fp8)
    # Optional: validate quantization
    # ref_input_fp8, _ = per_channel_quantize(input_t, quant_axis=1)
    # trt_input_fp8 = out_fp8_dict["input_fp8"]
    # print("Input quantization max diff:", torch.max(torch.abs(ref_input_fp8 - trt_input_fp8)).item())

    # Compute metrics
    diff = out_fp8 - out_fp32
    mse = torch.mean(diff**2).item()
    mae = torch.mean(torch.abs(diff)).item()
    rel_err = torch.mean(torch.abs(diff) / (torch.abs(out_fp32) + 1e-12)).item() * 100

    print("\n" + "=" * 50)
    print(f"FP8 vs FP32 MoE Error Metrics")
    print("=" * 50)
    print(f"MSE         : {mse:.6e}")
    print(f"MAE         : {mae:.6e}")
    print(f"RelErr (%)  : {rel_err:.4f}%")
