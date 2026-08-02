# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Dense Qwen3.5/3.6 checkpoints (e.g. Qwen3.6-27B-NVFP4) must reuse the same
ModelOpt NVFP4 / FP8 machinery as the MoE ones.

Two things are covered here:

* ``Qwen35TextModel`` -- not just ``Qwen35MoeTextModel`` -- owns the mapping from a
  dense ``mlp.{gate,up,down}_proj`` MatMul to its checkpoint key, so the plain
  decoder emits ``MatMulBlockQuantizedFp4Weight`` instead of falling back to INT4.
* ``ModeloptModel._build_layer`` populates ``layer.mlp.{gate,up,down}_proj`` for a
  dense checkpoint and the MoE router / shared-expert namespace otherwise.
"""

import onnx_ir as ir
import pytest

from models.builders.qwen import Qwen35MoeTextModel, Qwen35TextModel
from models.quantized_model import ModeloptModel


def _make_model(cls=Qwen35TextModel, **attrs):
    """Minimal builder stub exposing only the NVFP4 key mapping."""
    model = object.__new__(cls)
    model.is_mtp_head = False
    model.nvfp4_dense_exclude_layers = set()
    model.nvfp4_lmhead_fp16 = False
    model.use_original_nvfp4_weights = True
    for key, value in attrs.items():
        setattr(model, key, value)
    return model


DENSE_GATE = "/model/layers.3/mlp/gate_proj/MatMul"
DENSE_UP = "/model/layers.3/mlp/up_proj/MatMul"
DENSE_DOWN = "/model/layers.3/mlp/down_proj/MatMul"
SHARED_GATE = "/model/layers.3/shared_expert/gate_proj/MatMul"


@pytest.mark.parametrize("cls", [Qwen35TextModel, Qwen35MoeTextModel])
def test_dense_mlp_maps_to_checkpoint_keys(cls):
    model = _make_model(cls)

    assert model._nvfp4_dense_key_for_matmul(DENSE_GATE) == "model.language_model.layers.3.mlp.gate_proj"
    assert model._nvfp4_dense_key_for_matmul(DENSE_UP) == "model.language_model.layers.3.mlp.up_proj"
    assert model._nvfp4_dense_key_for_matmul(DENSE_DOWN) == "model.language_model.layers.3.mlp.down_proj"


def test_shared_expert_and_lm_head_still_map():
    model = _make_model()

    assert (
        model._nvfp4_dense_key_for_matmul(SHARED_GATE)
        == "model.language_model.layers.3.mlp.shared_expert.gate_proj"
    )
    assert model._nvfp4_dense_key_for_matmul("/lm_head/MatMul") == "lm_head"


def test_unrelated_matmuls_are_not_nvfp4():
    model = _make_model()

    for basename in (
        "/model/layers.3/attn/q_proj/MatMul",
        "/model/layers.3/linear_attn/in_proj_qkv/MatMul",
        "/model/layers.3/moe/router/MatMul",
    ):
        assert model._nvfp4_dense_key_for_matmul(basename) is None


def test_mtp_head_keeps_only_the_lm_head_in_nvfp4():
    model = _make_model(is_mtp_head=True)

    assert model._nvfp4_dense_key_for_matmul("/lm_head/MatMul") == "lm_head"
    assert model._nvfp4_dense_key_for_matmul(DENSE_GATE) is None


def test_dense_mlp_and_lm_head_are_excluded_from_int4_fallback():
    model = object.__new__(Qwen35TextModel)
    model.onnx_dtype = ir.DataType.INT4
    model.num_layers = 2
    model.quant_attrs = {}
    model._init_original_checkpoint_weight_options(
        {
            "use_original_nvfp4_weights": "true",
            "use_original_fp8_weights": "true",
            "nvfp4_dense_exclude_layers": "1",
            "nvfp4_lmhead_fp16": "true",
        }
    )

    excluded = set(model.quant_attrs["nodes_to_exclude"])
    # Layer 1 is excluded from NVFP4, so its dense MLP must not be quantized to INT4 either.
    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert f"/model/layers.1/mlp/{proj}/MatMul" in excluded
        assert f"/model/layers.1/shared_expert/{proj}/MatMul" in excluded
    # The BF16 linear-attention projections are never INT4.
    assert "/model/layers.0/linear_attn/in_proj_a/MatMul" in excluded
    assert "/model/layers.0/linear_attn/in_proj_b/MatMul" in excluded
    # nvfp4_lmhead_fp16 keeps the lm_head in fp16 rather than INT4.
    assert "/lm_head/MatMul" in excluded


class _StubModelopt(ModeloptModel):
    """`_build_layer` exercised against an in-memory tensor name set."""

    def __init__(self, names):
        from types import SimpleNamespace

        self._names = set(names)
        self._simple_namespace = SimpleNamespace

    def _get(self, name):
        return 1.0 if name in self._names else None

    def _dequant_linear(self, base):
        return 1.0 if f"{base}.weight" in self._names else None


def _layer_names(prefix, mlp_names):
    base = [
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.post_attention_layernorm.weight",
        f"{prefix}.self_attn.q_proj.weight",
        f"{prefix}.self_attn.k_proj.weight",
        f"{prefix}.self_attn.v_proj.weight",
        f"{prefix}.self_attn.o_proj.weight",
    ]
    return base + [f"{prefix}.{name}" for name in mlp_names]


def test_build_layer_populates_dense_mlp():
    prefix = "model.language_model.layers.0"
    names = _layer_names(
        prefix, ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
    )

    layer = _StubModelopt(names)._build_layer(0)

    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert getattr(layer.mlp, proj) is not None
        assert getattr(layer.mlp, proj).bias is None
    assert not hasattr(layer.mlp, "shared_expert")


def test_build_layer_populates_moe_mlp():
    prefix = "model.language_model.layers.0"
    names = _layer_names(
        prefix,
        [
            "mlp.gate.weight",
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
            "mlp.shared_expert.down_proj.weight",
        ],
    )

    layer = _StubModelopt(names)._build_layer(0)

    assert not hasattr(layer.mlp, "gate_proj")
    assert layer.mlp.shared_expert.gate_proj is not None
    assert layer.mlp.experts is None


def test_lazy_linear_defers_dequantization():
    """`_linear_module` must not dequantize until ``weight`` is read."""
    calls = []

    class _Counting(_StubModelopt):
        def _dequant_linear(self, base):
            calls.append(base)
            return 1.0

    module = _Counting({"x.weight"})._linear_module("x")
    assert calls == []
    assert module.weight == 1.0
    assert calls == ["x"]
    # Cached: a second read does not re-dequantize.
    assert module.weight == 1.0
    assert calls == ["x"]
