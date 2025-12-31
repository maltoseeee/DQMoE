import tensorrt as trt
import numpy as np
import ctypes
import torch
import os
from util import *

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
        "input", trt.DataType.FP8, (max_num_tokens, hidden_size)
    )
    weight1_cm = network.add_input(
        "weight1", trt.DataType.FP8, (num_experts, inter_size * 2, hidden_size)
    )
    weight2_cm = network.add_input(
        "weight2", trt.DataType.FP8, (num_experts, hidden_size, inter_size)
    )
    index_tensor = network.add_input(
        "index", trt.DataType.INT32, (max_num_tokens, top_k)
    )
    actscale = network.add_input("actscale", trt.DataType.FLOAT, (max_num_tokens, 1))
    scale_a = network.add_input(
        "scale_a",
        trt.DataType.FLOAT,
        (num_experts, inter_size * 2, 1),
    )
    scale_b = network.add_input(
        "scale_b",
        trt.DataType.FLOAT,
        (num_experts, hidden_size, 1),
    )
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
        input_tensor,
        weight1_cm,
        weight2_cm,
        index_tensor,
        bias1,
        bias2,
        actscale,
        scale_a,
        scale_b,
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


def build_block_fp8_engine():
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
        "input", trt.DataType.FP8, (max_num_tokens, hidden_size)
    )
    weight1_cm = network.add_input(
        "weight1", trt.DataType.FP8, (num_experts, inter_size * 2, hidden_size)
    )
    weight2_cm = network.add_input(
        "weight2", trt.DataType.FP8, (num_experts, hidden_size, inter_size)
    )
    index_tensor = network.add_input(
        "index", trt.DataType.INT32, (max_num_tokens, top_k)
    )

    def block_size(size):
        return (size + 127) // 128

    actscale = network.add_input(
        "actscale", trt.DataType.FLOAT, (max_num_tokens, block_size(hidden_size))
    )
    scale_a = network.add_input(
        "scale_a",
        trt.DataType.FLOAT,
        (num_experts, inter_size * 2, block_size(hidden_size)),
    )
    scale_b = network.add_input(
        "scale_b",
        trt.DataType.FLOAT,
        (num_experts, hidden_size, block_size(inter_size)),
    )

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
            "quant_mode", np.array([1 << 10], np.int32), trt.PluginFieldType.INT32
        ),
    ]
    plugin = creator.create_plugin(
        "mixture_of_experts", trt.PluginFieldCollection(fields)
    )

    inputs = [
        input_tensor,
        weight1_cm,
        weight2_cm,
        index_tensor,
        bias1,
        bias2,
        actscale,
        scale_a,
        scale_b,
    ]
    layer = network.add_plugin_v2(inputs, plugin)

    layer = network.add_cast(layer.get_output(0), trt.DataType.FLOAT)
    layer.get_output(0).name = "output"
    network.mark_output(layer.get_output(0))

    engine = builder.build_serialized_network(network, config)
    assert engine, "FP8 engine build failed!"
    if isinstance(engine, trt.ICudaEngine):
        engine = engine.serialize()
    with open("test_block_fp8.engine", "wb") as f:
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
        "weight1", trt.DataType.FLOAT, (num_experts, inter_size * 2, hidden_size)
    )
    weight2 = network.add_input(
        "weight2", trt.DataType.FLOAT, (num_experts, hidden_size, inter_size)
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

def build_fp16_engine():
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    config = builder.create_builder_config()

    input_tensor = network.add_input(
        "input", trt.DataType.HALF, (max_num_tokens, hidden_size)
    )
    weight1 = network.add_input(
        "weight1", trt.DataType.HALF, (num_experts, inter_size * 2, hidden_size)
    )
    weight2 = network.add_input(
        "weight2", trt.DataType.HALF, (num_experts, hidden_size, inter_size)
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
            np.array([int(trt.DataType.HALF)], np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "output_type",
            np.array([int(trt.DataType.HALF)], np.int32),
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
    layer = network.add_cast(layer.get_output(0), trt.DataType.FLOAT)
    layer.get_output(0).name = "output"
    network.mark_output(layer.get_output(0))

    engine = builder.build_serialized_network(network, config)
    assert engine, "FP16 engine build failed!"
    if isinstance(engine, trt.ICudaEngine):
        engine = engine.serialize()
    with open("test_fp16.engine", "wb") as f:
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


# === Main ===
if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = "cuda"

    # Build engines only if not exists
    build_engine_if_not_exists("test_fp32.engine", build_fp32_engine)
    build_engine_if_not_exists("test_fp16.engine", build_fp16_engine)
    build_engine_if_not_exists("test_fp8.engine", build_fp8_engine)
    build_engine_if_not_exists("test_block_fp8.engine", build_block_fp8_engine)

    # Generate data
    std = 0.05
    input_t = torch.normal(0, std, (max_num_tokens, hidden_size), device=device)
    weight1_t = torch.normal(
        0, std, (num_experts, inter_size * 2, hidden_size), device=device
    )
    weight2_t = torch.normal(
        0, std, (num_experts, hidden_size, inter_size), device=device
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

    input_dict_fp16 = {
        "input": input_t.half(),
        "weight1": weight1_t.half(),
        "weight2": weight2_t.half(),
        "index": index_t,
    }

    c_a_fp8 = per_token_cast_to_fp8(input_t)
    c_b_fp8 = per_channel_cast_to_fp8(weight1_t)
    c_b1_fp8 = per_channel_cast_to_fp8(weight2_t)
    # FP8 uses transposed layout (ColumnMajor)
    input_dict_fp8 = {
        "input": c_a_fp8[0],
        "weight1": c_b_fp8[0],
        "weight2": c_b1_fp8[0],
        "index": index_t,
        "actscale": c_a_fp8[1],
        "scale_a": c_b_fp8[1],
        "scale_b": c_b1_fp8[1],
    }

    a_fp8 = per_token_block_cast_to_fp8(input_t)
    b_fp8 = per_block_cast_to_fp8(weight1_t)
    b1_fp8 = per_block_cast_to_fp8(weight2_t)
    input_dict_block_fp8 = {
        "input": a_fp8[0],
        "actscale": a_fp8[1],
        "weight1": b_fp8[0],
        "scale_a": b_fp8[1],
        "weight2": b1_fp8[0],
        "scale_b": b1_fp8[1],
        "index": index_t,
    }

    # Inference
    out_fp32 = infer_trt_torch("test_fp32.engine", input_dict_fp32)["output"]
    out_fp8_dict = infer_trt_torch("test_fp8.engine", input_dict_fp8)
    out_fp8 = out_fp8_dict["output"]
    out_block_fp8 = infer_trt_torch("test_block_fp8.engine", input_dict_block_fp8)[
        "output"
    ]
    out_fp16 = infer_trt_torch("test_fp16.engine", input_dict_fp16)["output"]
    print(out_fp32)
    print(out_fp16)
    print(out_fp8)
    print(out_block_fp8)

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
