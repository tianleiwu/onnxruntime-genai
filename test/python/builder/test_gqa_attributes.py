from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parents[1]))


def _load_builder_module(module_name):
    spec = importlib.util.spec_from_file_location(f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("models", types.ModuleType("models"))
builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
builders_package.__path__ = [str(BUILDERS_DIR)]

base_module = _load_builder_module("base")
Model = base_module.Model


class _FakeGQAModel:
    is_fused_qk_norm_gqa_supported = Model.is_fused_qk_norm_gqa_supported
    make_quantized_kv_cache_init = Model.make_quantized_kv_cache_init
    make_group_query_attention = Model.make_group_query_attention

    def __init__(self, ep="cpu", fuse_qk_norm_gqa=True):
        self.ep = ep
        self.extra_options = {"fuse_qk_norm_gqa": fuse_qk_norm_gqa}
        self.num_attn_heads = 8
        self.num_kv_heads = 2
        self.head_size = 16
        self.window_size = -1
        self.attention_attrs = {
            "op_type": "GroupQueryAttention",
            "scale": 0.125,
            "softcap": 0.0,
            "rope": True,
            "use_rope_in_attn": True,
            "qk_norm_epsilon": 1e-6,
        }
        self.rope_attrs = {"interleaved": 0}
        self.io_dtype = None
        self.kv_cache_quant_type = "none"
        self.nodes = []

    def make_node(self, op_type, inputs, outputs, name, domain, **attributes):
        self.nodes.append({"inputs": inputs, "attributes": attributes})

    def make_value(self, *args, **kwargs):
        pass


def test_cpu_does_not_enable_fused_qk_norm_gqa_by_default():
    assert not _FakeGQAModel("cpu").is_fused_qk_norm_gqa_supported()
    assert _FakeGQAModel("cuda").is_fused_qk_norm_gqa_supported()


def test_plain_gqa_emits_qk_norm_epsilon_attribute():
    model = _FakeGQAModel()

    model.make_group_query_attention("/gqa", q_path="q", k_path="k", v_path="v")

    assert model.nodes[-1]["attributes"]["qk_norm_epsilon"] == 1e-6


def test_fused_qk_norm_gqa_emits_qk_norm_epsilon_attribute():
    model = _FakeGQAModel("cuda")

    model.make_group_query_attention(
        "/gqa",
        q_path="q",
        k_path="k",
        v_path="v",
        q_norm_weight="q_norm_weight",
        k_norm_weight="k_norm_weight",
    )

    assert model.nodes[-1]["attributes"]["qk_norm_epsilon"] == 1e-6


def test_quantized_gqa_wires_scales_before_qk_norm_inputs():
    model = _FakeGQAModel("cuda")
    model.kv_cache_quant_type = "int8_per_channel"
    model.kv_quant_type = "PER_CHANNEL"
    model.kv_cache_bit_width = 8

    model.make_group_query_attention(
        "/gqa",
        layer_id=3,
        q_path="q",
        k_path="k",
        v_path="v",
        q_norm_weight="q_norm_weight",
        k_norm_weight="k_norm_weight",
    )

    node = model.nodes[-1]
    assert node["inputs"][12:16] == [
        "/model/kv_cache_scales/k_scale.3",
        "/model/kv_cache_scales/v_scale.3",
        "q_norm_weight",
        "k_norm_weight",
    ]
    assert node["attributes"]["k_quant_type"] == "PER_CHANNEL"
    assert node["attributes"]["v_quant_type"] == "PER_CHANNEL"
    assert node["attributes"]["kv_cache_bit_width"] == 8


def test_int4_kv_cache_uses_uint8_packed_head_dimension():
    model = _FakeGQAModel("cuda")
    model.kv_cache_quant_type = "int4_per_tensor"
    model.head_size = 17
    model.input_types = {"past_key_values.key": None, "past_key_values.value": None}
    model.output_types = {"present.key": None, "present.value": None}
    model.input_shapes = {
        "past_key_values.key": ["batch", "heads", "past", 17],
        "past_key_values.value": ["batch", "heads", "past", 17],
    }
    model.output_shapes = {
        "present.key": ["batch", "heads", "total", 17],
        "present.value": ["batch", "heads", "total", 17],
    }
    model.past_present_share_buffer = True

    model.make_quantized_kv_cache_init()

    assert model.input_types["past_key_values.key"] == base_module.ir.DataType.UINT8
    assert model.output_types["present.value"] == base_module.ir.DataType.UINT8
    assert model.input_shapes["past_key_values.key"][-1] == 9
    assert model.output_shapes["present.value"][-1] == 9
    assert model.past_present_share_buffer

    model.ep = "cpu"
    model.make_quantized_kv_cache_init()
    assert not model.past_present_share_buffer


def test_fp8_kv_cache_uses_float8_dtype_without_packing():
    model = _FakeGQAModel("cuda")
    model.kv_cache_quant_type = "fp8_per_tensor"
    model.input_types = {"past_key_values.key": None, "past_key_values.value": None}
    model.output_types = {"present.key": None, "present.value": None}
    model.input_shapes = {
        "past_key_values.key": ["batch", "heads", "past", 64],
        "past_key_values.value": ["batch", "heads", "past", 64],
    }
    model.output_shapes = {
        "present.key": ["batch", "heads", "total", 64],
        "present.value": ["batch", "heads", "total", 64],
    }

    model.make_quantized_kv_cache_init()

    assert model.input_types["past_key_values.key"] == base_module.ir.DataType.FLOAT8E4M3FN
    assert model.output_types["present.value"] == base_module.ir.DataType.FLOAT8E4M3FN
    assert model.kv_cache_bit_width == 8
    assert model.kv_quant_type == "PER_TENSOR"
    # FP8 is not bit-packed: the head dimension is unchanged.
    assert model.input_shapes["past_key_values.key"][-1] == 64
    assert model.output_shapes["present.value"][-1] == 64
