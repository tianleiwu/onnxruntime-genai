# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from types import MethodType

import onnx_ir as ir
import torch

from models.builders.qwen import Qwen35MoeTextModel


def _make_model(scales, use_static_scale=True):
    model = object.__new__(Qwen35MoeTextModel)
    model.io_dtype = ir.DataType.FLOAT16
    model.fp8_attn_static_input_scale = use_static_scale
    model.share_fp8_attn_qkv_activation = True
    model._fp8_attention_activation_cache = {}
    model._fp8_weight_key_for_matmul = MethodType(lambda self, basename: basename, model)
    model._load_nvfp4_tensor = MethodType(
        lambda self, key: torch.tensor(scales[key.removesuffix(".input_scale")]), model
    )

    emitted_nodes = []

    def record_node(self, *args, **kwargs):
        emitted_nodes.append((args, kwargs))

    for method_name in (
        "make_add",
        "make_cast",
        "make_clip",
        "make_constant_of_shape",
        "make_div",
        "make_mul",
        "make_node",
        "make_reduce_max",
        "make_reshape",
        "make_value",
    ):
        setattr(model, method_name, MethodType(record_node, model))
    return model, emitted_nodes


def test_static_fp8_activation_is_shared_for_matching_qkv_inputs_and_scales():
    model, emitted_nodes = _make_model({"q": 0.125, "k": 0.125, "v": 0.125})

    q_activation = model._make_fp8_attention_activation("q", "hidden", 2048, "sequence_length")
    nodes_after_q = len(emitted_nodes)
    k_activation = model._make_fp8_attention_activation("k", "hidden", 2048, "sequence_length")
    v_activation = model._make_fp8_attention_activation("v", "hidden", 2048, "sequence_length")

    assert k_activation is q_activation
    assert v_activation is q_activation
    assert len(emitted_nodes) == nodes_after_q


def test_static_fp8_activation_does_not_emit_dynamic_amax_path():
    model, emitted_nodes = _make_model({"q": 0.125})

    model._make_fp8_attention_activation("q", "hidden", 2048, "sequence_length")

    emitted_op_types = [args[0] for args, _ in emitted_nodes if args and isinstance(args[0], str)]
    assert "Abs" not in emitted_op_types
    assert "Shape" in emitted_op_types
    assert any("value" in kwargs and kwargs.get("dtype") == ir.DataType.FLOAT for _, kwargs in emitted_nodes)
    assert all("Amax" not in str(args) for args, _ in emitted_nodes)
    assert all("ScaleFloor" not in str(args) for args, _ in emitted_nodes)


def test_dynamic_fp8_activation_keeps_amax_path():
    model, emitted_nodes = _make_model({}, use_static_scale=False)

    model._make_fp8_attention_activation("q", "hidden", 2048, "sequence_length")

    emitted_op_types = [args[0] for args, _ in emitted_nodes if args and isinstance(args[0], str)]
    assert "Abs" in emitted_op_types
    assert any("Amax" in str(args) for args, _ in emitted_nodes)
    assert any("ScaleFloor" in str(args) for args, _ in emitted_nodes)
    assert all("value" not in kwargs for _, kwargs in emitted_nodes)


def test_static_fp8_activation_is_not_shared_when_scale_differs():
    model, emitted_nodes = _make_model({"q": 0.125, "o": 0.25})

    q_activation = model._make_fp8_attention_activation("q", "hidden", 2048, "sequence_length")
    nodes_after_q = len(emitted_nodes)
    o_activation = model._make_fp8_attention_activation("o", "hidden", 2048, "sequence_length")

    assert o_activation is not q_activation
    assert len(emitted_nodes) > nodes_after_q