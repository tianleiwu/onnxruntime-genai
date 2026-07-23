// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "env_utils.h"
#include "model.h"
#include "kv_cache.h"  // For ComposeKeyValueName
#include "recurrent_state.h"
#include <algorithm>

namespace Generators {

RecurrentState::RecurrentState(State& state)
    : state_{state} {
  const auto& past_key_template = model_.config_->model.decoder.inputs.past_key_names;
  const auto& present_key_template = model_.config_->model.decoder.outputs.present_key_names;

  // Derive recurrent name templates from KV name templates
  auto derive_template = [](const std::string& kv_template, const std::string& suffix) -> std::string {
    auto pos = kv_template.rfind('.');
    if (pos == std::string::npos) return "";
    return kv_template.substr(0, pos + 1) + suffix;
  };

  std::string past_conv_template = derive_template(past_key_template, "conv_state");
  std::string past_recurrent_template = derive_template(past_key_template, "recurrent_state");
  std::string present_conv_template = derive_template(present_key_template, "conv_state");
  std::string present_recurrent_template = derive_template(present_key_template, "recurrent_state");

  if (past_conv_template.empty()) return;

  // Discover recurrent layer indices by scanning all session input names
  for (const auto& name : model_.session_info_.GetInputNames()) {
    // Try to match against the conv_state template (e.g. "past_key_values.%d.conv_state")
    // Extract the layer index from names that match
    auto prefix = past_conv_template.substr(0, past_conv_template.find('%'));
    auto suffix = past_conv_template.substr(past_conv_template.find('%') + 2);  // skip %d
    if (name.size() > prefix.size() + suffix.size() &&
        name.compare(0, prefix.size(), prefix) == 0 &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      auto idx_str = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
      int idx = std::stoi(idx_str);
      layer_indices_.push_back(idx);
    }
  }
  std::sort(layer_indices_.begin(), layer_indices_.end());

  if (layer_indices_.empty()) return;

  if (g_log.enabled)
    Log("info", "RecurrentState: Auto-discovered " + std::to_string(layer_indices_.size()) + " recurrent layers (indices: " + [&]() {
                      std::string s;
                      for (size_t i = 0; i < layer_indices_.size(); ++i) {
                        if (i) s += ",";
                        s += std::to_string(layer_indices_[i]);
                      }
                      return s; }() + ")");

  for (int idx : layer_indices_) {
    input_name_strings_.push_back(ComposeKeyValueName(past_conv_template, idx));
    input_name_strings_.push_back(ComposeKeyValueName(past_recurrent_template, idx));
    output_name_strings_.push_back(ComposeKeyValueName(present_conv_template, idx));
    output_name_strings_.push_back(ComposeKeyValueName(present_recurrent_template, idx));
  }

  conv_type_ = model_.session_info_.GetInputDataType(input_name_strings_[0]);
  recurrent_type_ = model_.session_info_.GetInputDataType(input_name_strings_[1]);

  auto fix_batch_dim = [&](std::vector<int64_t> shape) -> std::vector<int64_t> {
    if (!shape.empty() && shape[0] <= 0) {
      shape[0] = state_.params_->BatchBeamSize();
    }
    return shape;
  };

  conv_shape_ = fix_batch_dim(model_.session_info_.GetInputShape(input_name_strings_[0]));
  recurrent_shape_ = fix_batch_dim(model_.session_info_.GetInputShape(input_name_strings_[1]));

  // Validate all dims are positive (only batch dim is expected to be dynamic)
  auto validate_shape = [](const std::vector<int64_t>& shape, const std::string& name) {
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] <= 0)
        throw std::runtime_error("RecurrentState: " + name + " has unsupported dynamic dim " +
                                 std::to_string(shape[i]) + " at axis " + std::to_string(i));
    }
  };
  validate_shape(conv_shape_, "conv_state");
  validate_shape(recurrent_shape_, "recurrent_state");

  const int num_layers = static_cast<int>(layer_indices_.size());

  share_buffers_ = state_.params_->IsPastPresentShareBufferEnabled(model_.config_->model.type);

  // WebGPU prohibits binding the same buffer as both read-only (input) and
  // read-write (output) storage in the same compute pass, so it must use
  // separate past/present buffers with swap. All other EPs share buffers
  // for stable addresses (required by TRT-RTX graph replay, beneficial elsewhere).
  // TODO: Remove WebGPU special case once the ORT WebGPU EP adds a
  // LinearAttention kernel with native past/present buffer sharing support.
  const bool is_webgpu = model_.p_device_kvcache_->GetType() == DeviceType::WEBGPU;

  // Under CUDA-graph capture the recurrent (conv + linear-attention) state MUST be
  // double-buffered, not shared in place. Unlike GroupQueryAttention's KV share-buffer,
  // the LinearAttention / CausalConvWithState kernels update the recurrent state in place
  // (present_state aliased onto past_state); capturing that in-place update in a CUDA graph
  // produces a small but systematic per-step logit bias on replay that derails greedy
  // decoding (observed MMLU-Pro collapse ~85% -> ~21% with graph on). Double-buffering
  // (distinct past/present with a per-step swap and two captured graph variants) is proven
  // bit-faithful to eager, so make it the default whenever graph capture is enabled.
  // ORT_MTP_DOUBLE_BUFFER_RECURRENT_GRAPH forces the behavior on ("1") or off ("0").
  const std::string double_buffer_env = GetEnv("ORT_MTP_DOUBLE_BUFFER_RECURRENT_GRAPH");
  const bool double_buffer_default_on = state_.params_->use_graph_capture;
  const bool double_buffer_requested =
      double_buffer_env == "1" || (double_buffer_env != "0" && double_buffer_default_on);
  graph_double_buffer_ = !is_webgpu && double_buffer_requested;
  share_buffers_ = !is_webgpu && !graph_double_buffer_;

  if (!share_buffers_) {
    pasts_.resize(num_layers * 2);
  }
  presents_.reserve(num_layers * 2);

  auto& allocator = model_.p_device_kvcache_->GetAllocator();

  for (int i = 0; i < num_layers; ++i) {
    if (!share_buffers_) {
      pasts_[i * 2] = OrtValue::CreateTensor(allocator, conv_shape_, conv_type_);
      pasts_[i * 2 + 1] = OrtValue::CreateTensor(allocator, recurrent_shape_, recurrent_type_);
    }
    presents_.push_back(OrtValue::CreateTensor(allocator, conv_shape_, conv_type_));
    presents_.push_back(OrtValue::CreateTensor(allocator, recurrent_shape_, recurrent_type_));
  }

  if (!share_buffers_) {
    ZeroStates(pasts_);
  }
  ZeroStates(presents_);

  // Discover the optional per-position state outputs (present_state_all), emitted when the model
  // is built with emit_recurrent_state_all=true. They enable lossless multi-token MTP by letting
  // the controller crop the recurrent state to the accepted length (copy present_state_all[:, a]
  // into the live state) instead of a full main-model replay forward.
  std::string present_conv_all_template = derive_template(present_key_template, "conv_state_all");
  std::string present_recurrent_all_template = derive_template(present_key_template, "recurrent_state_all");
  if (!present_recurrent_all_template.empty()) {
    const std::string probe = ComposeKeyValueName(present_recurrent_all_template, layer_indices_[0]);
    has_state_all_ = model_.session_info_.HasOutput(probe);
  }

  if (has_state_all_) {
    const std::string binding = GetEnv("ORT_MTP_STATE_ALL_BINDING");
    if (binding == "conv") {
      bind_recurrent_all_ = false;
    } else if (binding == "recurrent") {
      bind_conv_all_ = false;
    } else if (binding == "none") {
      bind_conv_all_ = false;
      bind_recurrent_all_ = false;
    } else if (!binding.empty() && binding != "all") {
      throw std::runtime_error("ORT_MTP_STATE_ALL_BINDING must be all, conv, recurrent, or none");
    }

    // Per-position shapes: insert the seq_len axis at position 1 of the live-state shapes.
    conv_all_shape_ = {conv_shape_[0], 0, conv_shape_[1], conv_shape_[2]};
    recurrent_all_shape_ = {recurrent_shape_[0], 0, recurrent_shape_[1], recurrent_shape_[2], recurrent_shape_[3]};
    presents_all_.reserve(num_layers * 2);
    for (int i = 0; i < num_layers; ++i) {
      output_all_name_strings_.push_back(ComposeKeyValueName(present_conv_all_template, layer_indices_[i]));
      output_all_name_strings_.push_back(ComposeKeyValueName(present_recurrent_all_template, layer_indices_[i]));
      presents_all_.push_back(std::make_unique<Tensor>(model_.p_device_kvcache_, conv_type_));
      presents_all_.push_back(std::make_unique<Tensor>(model_.p_device_kvcache_, recurrent_type_));
    }
  }
}

void RecurrentState::Add() {
  if (layer_indices_.empty()) return;

  input_index_ = state_.inputs_.size();
  output_index_ = state_.outputs_.size();

  const int num_layers = static_cast<int>(layer_indices_.size());
  for (int i = 0; i < num_layers * 2; ++i) {
    // Shared buffers: alias input=output for stable addresses (required for graph capture).
    // Separate buffers: use distinct past/present allocations with per-step pointer swap.
    state_.inputs_.push_back(share_buffers_ ? presents_[i].get() : pasts_[i].get());
    state_.input_names_.push_back(input_name_strings_[i].c_str());
    state_.outputs_.push_back(presents_[i].get());
    state_.output_names_.push_back(output_name_strings_[i].c_str());
  }

  // Register the per-position state outputs as managed outputs (static-buffer, so they survive
  // CUDA-graph capture). Their OrtValue is (re)created per step by UpdateAll(); push nullptr here.
  if (has_state_all_) {
    output_all_index_ = state_.outputs_.size();
    for (int i = 0; i < num_layers; ++i) {
      if (bind_conv_all_) {
        state_.outputs_.push_back(presents_all_[i * 2]->GetOrtTensor());  // nullptr until UpdateAll()
        state_.output_names_.push_back(output_all_name_strings_[i * 2].c_str());
      }
      if (bind_recurrent_all_) {
        state_.outputs_.push_back(presents_all_[i * 2 + 1]->GetOrtTensor());
        state_.output_names_.push_back(output_all_name_strings_[i * 2 + 1].c_str());
      }
    }
  }
}

void RecurrentState::UpdateAll(int sequence_length) {
  if (!has_state_all_) return;
  // Only rebuild when the sequence length changes (matches HiddenStatesOutputs). conv_all_shape_[1]
  // and recurrent_all_shape_[1] track together.
  if (static_cast<int64_t>(sequence_length) == conv_all_shape_[1]) return;

  const int max_cap = state_.params_->max_graph_capture_length;
  // Static buffer when graph capture is active so the captured graph binds a stable output address.
  // Pre-size to the max captured length (the N+1-token MTP verify shape).
  const bool use_static = state_.params_->use_graph_capture && sequence_length >= 1 && sequence_length <= max_cap;
  const int num_layers = static_cast<int>(layer_indices_.size());

  conv_all_shape_[1] = sequence_length;
  recurrent_all_shape_[1] = sequence_length;

  const size_t conv_pos_elems = static_cast<size_t>(conv_all_shape_[2]) * conv_all_shape_[3];
  const size_t rec_pos_elems = static_cast<size_t>(recurrent_all_shape_[2]) * recurrent_all_shape_[3] * recurrent_all_shape_[4];

  size_t output_index = output_all_index_;
  for (int i = 0; i < num_layers; ++i) {
    if (bind_conv_all_) {
      const size_t conv_cap = use_static ? static_cast<size_t>(conv_all_shape_[0]) * max_cap * conv_pos_elems * Ort::SizeOf(conv_type_) : 0;
      presents_all_[i * 2]->CreateTensor(conv_all_shape_, use_static, conv_cap);
      state_.outputs_[output_index++] = presents_all_[i * 2]->GetOrtTensor();
    }

    if (bind_recurrent_all_) {
      const size_t rec_cap = use_static ? static_cast<size_t>(recurrent_all_shape_[0]) * max_cap * rec_pos_elems * Ort::SizeOf(recurrent_type_) : 0;
      presents_all_[i * 2 + 1]->CreateTensor(recurrent_all_shape_, use_static, rec_cap);
      state_.outputs_[output_index++] = presents_all_[i * 2 + 1]->GetOrtTensor();
    }
  }
}

void RecurrentState::CropToPosition(size_t position) {
  if (!HasStateAll())
    throw std::runtime_error("RecurrentState::CropToPosition requires the model exported with emit_recurrent_state_all=true");
  // Copy present_state_all[:, position] (the recurrent/conv state AFTER token `position` of the
  // last forward) into the live present buffers. Assumes batch_size == 1 (MTP is batch 1): each
  // per-position slice is a contiguous block of size == the live per-layer state.
  auto& device = *model_.p_device_;
  for (size_t j = 0; j < presents_.size(); ++j) {
    auto dst = ByteWrapTensor(device, *presents_[j]);
    auto all = ByteWrapTensor(device, *presents_all_[j]->GetOrtTensor());
    const size_t slice_bytes = dst.size();
    dst.CopyFrom(all.subspan(position * slice_bytes, slice_bytes));
  }
}

void RecurrentState::Update() {
  if (layer_indices_.empty() || share_buffers_) return;

  const int num_layers = static_cast<int>(layer_indices_.size());
  for (int i = 0; i < num_layers * 2; ++i) {
    std::swap(pasts_[i], presents_[i]);
    state_.inputs_[input_index_ + i] = pasts_[i].get();
    state_.outputs_[output_index_ + i] = presents_[i].get();
  }
  if (graph_double_buffer_) graph_buffer_variant_ ^= 1;
}

void RecurrentState::RewindTo(size_t index) {
  if (layer_indices_.empty()) return;

  if (index != 0) {
    // The recurrent (conv + linear-attention) state cannot be cropped like the
    // attention KV cache. A partial rewind is only possible by restoring a
    // snapshot captured at the target length (used by speculative decoding to
    // roll back a rejected draft). The caller (e.g. the MTP orchestrator)
    // guarantees the snapshot was taken at exactly `index`.
    if (snapshot_valid_) {
      RestoreSnapshot();
      return;
    }
    throw std::runtime_error(
        "RecurrentState::RewindTo(" + std::to_string(index) +
        ") is not supported without a snapshot. Recurrent states cannot be partially rewound; "
        "call Snapshot() at the target length first (e.g. for speculative decoding).");
  }
  // Full reset to length 0.
  snapshot_valid_ = false;
  if (share_buffers_) {
    // Shared buffers: zero in place, addresses stay stable.
    ZeroStates(presents_);
  } else {
    // Zero and rebind all state buffers.
    ZeroStates(pasts_);
    ZeroStates(presents_);
    const int num_layers = static_cast<int>(layer_indices_.size());
    for (int i = 0; i < num_layers * 2; ++i) {
      state_.inputs_[input_index_ + i] = pasts_[i].get();
      state_.outputs_[output_index_ + i] = presents_[i].get();
    }
  }
}

void RecurrentState::ZeroStates(std::vector<std::unique_ptr<OrtValue>>& states) {
  auto& device = *model_.p_device_kvcache_;
  for (auto& val : states) {
    ByteWrapTensor(device, *val).Zero();
  }
}

void RecurrentState::CopyStates(const std::vector<std::unique_ptr<OrtValue>>& src,
                                std::vector<std::unique_ptr<OrtValue>>& dst) {
  auto& device = *model_.p_device_kvcache_;
  for (size_t i = 0; i < src.size(); ++i) {
    ByteWrapTensor(device, *dst[i]).CopyFrom(ByteWrapTensor(device, *src[i]));
  }
}

void RecurrentState::Snapshot() {
  if (layer_indices_.empty()) return;

  // The live state is in presents_ (shared-buffer EPs alias input==output to it;
  // non-shared EPs keep the latest in presents_ after Update()'s swap).
  if (snapshot_.empty()) {
    auto& allocator = model_.p_device_kvcache_->GetAllocator();
    const int num_layers = static_cast<int>(layer_indices_.size());
    snapshot_.reserve(num_layers * 2);
    for (int i = 0; i < num_layers; ++i) {
      snapshot_.push_back(OrtValue::CreateTensor(allocator, conv_shape_, conv_type_));
      snapshot_.push_back(OrtValue::CreateTensor(allocator, recurrent_shape_, recurrent_type_));
    }
  }
  CopyStates(presents_, snapshot_);
  snapshot_valid_ = true;
}

void RecurrentState::RestoreSnapshot() {
  if (layer_indices_.empty()) return;
  if (!snapshot_valid_) {
    throw std::runtime_error("RecurrentState::RestoreSnapshot called before Snapshot");
  }
  // Copy back into the live buffers in place so their addresses stay stable
  // (required by CUDA-graph replay, which captures fixed buffer pointers).
  CopyStates(snapshot_, presents_);
}

std::unique_ptr<RecurrentState> CreateRecurrentState(State& state) {
  auto recurrent_state = std::make_unique<RecurrentState>(state);
  if (recurrent_state->IsEmpty()) {
    return nullptr;
  }
  return recurrent_state;
}

}  // namespace Generators
