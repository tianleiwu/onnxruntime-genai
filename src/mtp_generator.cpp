// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "search.h"
#include "constrained_logits_processor.h"
#include "models/model.h"
#include "mtp_generator.h"

#include <cstdlib>
#include <cstring>

namespace Generators {

namespace {
// Greedy argmax over a contiguous vocab row of fp32 logits on the CPU.
int32_t ArgmaxRow(const float* row, int vocab_size) {
  int32_t best = 0;
  float best_val = row[0];
  for (int i = 1; i < vocab_size; ++i) {
    if (row[i] > best_val) {
      best_val = row[i];
      best = i;
    }
  }
  return best;
}
}  // namespace

MtpGenerator::MtpGenerator(const Model& main_model, const Model& mtp_model, const GeneratorParams& params)
    : main_model_{main_model}, mtp_model_{mtp_model} {
  // Number of speculative draft tokens per step (N). N=1 is the original single-token fast path;
  // N>1 chains the single MTP module N times (feeding its own post-norm hidden back), as vLLM's
  // AutoRegressiveSpeculator does. Tunable via env var for benchmarking without an API change;
  // N>1 requires the head exported with `mtp_emit_hidden=true` (extra output hidden_states_out).
  num_speculative_tokens_ = 1;
  if (const char* env = std::getenv("ORT_MTP_NUM_SPECULATIVE_TOKENS")) {
    const int v = std::atoi(env);
    if (v >= 1) num_speculative_tokens_ = v;
  }
  // Capture the 1-token decode and the verify shapes up to N+1 tokens.
  auto& main_params = const_cast<GeneratorParams&>(params);
  main_params.max_graph_capture_length = num_speculative_tokens_ + 1;

  main_ = CreateGenerator(main_model_, params);
  mtp_params_ = std::make_shared<GeneratorParams>(mtp_model_);
  mtp_params_->search = params.search;
  mtp_params_->max_graph_capture_length = 1;
  mtp_params_->use_graph_capture = false;
  mtp_ = CreateGenerator(mtp_model_, *mtp_params_);

  hidden_size_ = main_model_.config_->model.decoder.hidden_size;
  vocab_size_ = main_model_.config_->model.vocab_size;
  max_length_ = params.search.max_length;

  // Reusable [1, 1, hidden] device buffer for the on-device hidden-state handoff.
  hidden_slice_ = std::make_shared<Tensor>(
      main_model_.p_device_inputs_,
      main_model_.session_info_.GetOutputDataType(main_model_.config_->model.decoder.outputs.hidden_states));
  const std::array<int64_t, 3> slice_shape{1, 1, hidden_size_};
  hidden_slice_->CreateTensor(slice_shape);

  // Reusable [1, 2, hidden] device buffer for the batched 2-token draft (post-accept KV-advance
  // fused with the next step's draft into one MTP forward).
  hidden_slice2_ = std::make_shared<Tensor>(
      main_model_.p_device_inputs_,
      main_model_.session_info_.GetOutputDataType(main_model_.config_->model.decoder.outputs.hidden_states));
  const std::array<int64_t, 3> slice2_shape{1, 2, hidden_size_};
  hidden_slice2_->CreateTensor(slice2_shape);

  // Multi-token (N>1) scratch: the head's own hidden output (chain feedback) and a re-feed buffer
  // used to re-materialize accepted drafts in the head KV with the main model's hidden states.
  if (num_speculative_tokens_ > 1) {
    head_out_hidden_ = std::make_shared<Tensor>(
      mtp_model_.p_device_inputs_,
      mtp_model_.session_info_.GetInputDataType(mtp_model_.config_->model.decoder.inputs.hidden_states));
    head_out_hidden_->CreateTensor(slice_shape);
    refeed_hidden_ = std::make_shared<Tensor>(
      mtp_model_.p_device_inputs_,
      mtp_model_.session_info_.GetInputDataType(mtp_model_.config_->model.decoder.inputs.hidden_states));
    refeed_hidden_->CreateTensor(slice_shape);
    drafts_.resize(num_speculative_tokens_);
    verify_tokens_.resize(num_speculative_tokens_ + 1);
    verify_argmax_.resize(num_speculative_tokens_ + 1);
  }
}

void MtpGenerator::ExtractHiddenPosition(OrtValue* hidden, int position) {
  // hidden is [1, S, H] on the model device; copy row `position` into hidden_slice_ ([1,1,H]).
  CopyHiddenRow(hidden, position, *hidden_slice_);
}

void MtpGenerator::CopyHiddenRow(OrtValue* hidden, int position, Tensor& dst) {
  // hidden is [1, S, H] on the main model device; copy row `position` into dst ([1,1,H]).
  auto src = ByteWrapTensor(*main_model_.p_device_, *hidden);
  const size_t row_bytes = dst.GetByteSpan().size();
  auto src_row = src.subspan(static_cast<size_t>(position) * row_bytes, row_bytes);
  dst.GetByteSpan().CopyFrom(src_row);
}

int32_t MtpGenerator::DraftHeadStep(int32_t token, bool need_draft) {
  // The head's `hidden_states` input must already be set by the caller (SetHiddenStates). Append
  // `token` (the head KV grows by one), capture the head's own post-final-norm output for the next
  // chained step, and return the greedy draft for the token after `token`.
  std::array<int32_t, 1> tok{token};
  mtp_->AppendTokens(cpu_span<const int32_t>(tok));
  ++head_len_;

  int32_t draft = 0;
  if (need_draft) {
    // ArgMax synchronizes the head's logits producer before the feedback D2D copy below.
    auto logits_span = mtp_->GetLogits();  // fp32, last token, [1, V]
    if (!mtp_model_.p_device_->ArgMax(logits_span.Span().data(), Ort::TypeToTensorType<float>, 1, vocab_size_, &draft)) {
      auto logits = logits_span.CopyDeviceToCpu();  // host fallback
      draft = ArgmaxRow(logits.data(), vocab_size_);
    }
  }

  // Capture the head's recurrent feedback hidden (hidden_states_out, the single processed row).
  OrtValue* head_hidden = mtp_->state_->GetOutput("hidden_states_out");
  if (head_hidden == nullptr) {
    throw std::runtime_error(
        "MtpGenerator: multi-token speculation requires the MTP head exported with "
        "mtp_emit_hidden=true (missing 'hidden_states_out' output).");
  }
  const size_t row_bytes = head_out_hidden_->GetByteSpan().size();
  // The head ran a single token, so hidden_states_out is [1,1,H]: take row 0.
  auto dst = head_out_hidden_->GetByteSpan();
  if (head_hidden->GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_CPU) {
    // ExtraOutputs are not IO-bound, so ORT may allocate this output on the CPU. Stage it through
    // the destination's pinned host buffer rather than passing a host pointer to cudaMemcpy D2D.
    auto dst_cpu = dst.CpuSpan();
    std::memcpy(dst_cpu.data(), head_hidden->GetTensorRawData(), row_bytes);
    dst.CopyCpuToDevice();
  } else {
    auto src = ByteWrapTensor(*mtp_model_.p_device_, *head_hidden);
    dst.CopyFrom(src.subspan(0, row_bytes));
  }
  return draft;
}

void MtpGenerator::ArgmaxMainRows(int first_row, int num_rows, int32_t* out) {
  OrtValue* raw = main_->state_->GetOutput(main_model_.config_->model.decoder.outputs.logits.c_str());
  auto info = raw->GetTensorTypeAndShapeInfo();
  const ONNXTensorElementDataType type = info->GetElementType();

  if (std::getenv("ORT_MTP_LOG_TOP2_MARGINS") != nullptr) {
    std::vector<int32_t> top2_tokens(static_cast<size_t>(num_rows) * 2);
    std::vector<float> top2_scores(static_cast<size_t>(num_rows) * 2);
    const uint8_t* base = static_cast<const uint8_t*>(raw->GetTensorRawData());
    const void* rows = base + static_cast<size_t>(first_row) * vocab_size_ * Ort::SizeOf(type);
    if (main_model_.p_device_->Top2(rows, type, num_rows, vocab_size_, top2_tokens.data(), top2_scores.data())) {
      for (int row = 0; row < num_rows; ++row) {
        std::cout << "MTP_TOP2 row=" << first_row + row
                  << " top1=" << top2_tokens[row * 2]
                  << " top2=" << top2_tokens[row * 2 + 1]
                  << " margin=" << top2_scores[row * 2] - top2_scores[row * 2 + 1] << std::endl;
        out[row] = top2_tokens[row * 2];
      }
      return;
    }
  }

  // The CUDA distributed-select Top-K implementation is batch-1 only. The N=1 MTP verify uses
  // two rows and is covered by its existing tuned path, but N>1 verifies have 3+ rows. Submit
  // those rows independently so each invocation uses the proven batch-1 path while keeping the
  // full logits device-resident. This is a compatibility bridge until Top-K has a native batched
  // argmax path for the large Qwen vocabulary.
  if (num_rows > 2) {
    const uint8_t* base = static_cast<const uint8_t*>(raw->GetTensorRawData());
    const size_t row_bytes = static_cast<size_t>(vocab_size_) * Ort::SizeOf(type);
    bool all_device = true;
    for (int row = 0; row < num_rows; ++row) {
      const void* row_ptr = base + static_cast<size_t>(first_row + row) * row_bytes;
      if (!main_model_.p_device_->ArgMax(row_ptr, type, 1, vocab_size_, out + row)) {
        all_device = false;
        break;
      }
    }
    if (all_device) return;
  }

  // Fast path: argmax the rows on-device with the high-performance Top-K kernel (k=1). Only the
  // small token ids are copied to the host -- the full [1,S,V] logits never leave the GPU.
  const uint8_t* base = static_cast<const uint8_t*>(raw->GetTensorRawData());
  const void* row_ptr = base + static_cast<size_t>(first_row) * vocab_size_ * Ort::SizeOf(type);
  if (main_model_.p_device_->ArgMax(row_ptr, type, num_rows, vocab_size_, out))
    return;

  // Host fallback (e.g. CPU device): cast the logits to fp32, copy to the host, argmax each row.
  Cast(*raw, logits_fp32_, *main_model_.p_device_, Ort::TypeToTensorType<float>);
  auto span = ByteWrapTensor(*main_model_.p_device_, *logits_fp32_);
  auto cpu = span.CopyDeviceToCpu();
  const float* data = reinterpret_cast<const float*>(cpu.data());
  for (int r = 0; r < num_rows; ++r)
    out[r] = ArgmaxRow(data + static_cast<size_t>(first_row + r) * vocab_size_, vocab_size_);
}

int32_t MtpGenerator::DraftNextToken(OrtValue* /*unused*/, int32_t token, bool need_draft) {
  // hidden_slice_ already holds the hidden state paired with `token`. Feed (hidden, token) to the
  // MTP head; its KV cache accumulates, so this is an O(1) incremental draft step.
  mtp_->SetHiddenStates(hidden_slice_);
  std::array<int32_t, 1> tok{token};
  mtp_->AppendTokens(cpu_span<const int32_t>(tok));
  if (!need_draft) {
    // KV-advance only (e.g. after an accepted draft): skip the full-vocab argmax + stream sync.
    return 0;
  }
  auto logits_span = mtp_->GetLogits();              // fp32, last token, [1, V]
  int32_t draft = 0;
  if (mtp_model_.p_device_->ArgMax(logits_span.Span().data(), Ort::TypeToTensorType<float>, 1, vocab_size_, &draft))
    return draft;
  auto logits = logits_span.CopyDeviceToCpu();        // host fallback
  return ArgmaxRow(logits.data(), vocab_size_);
}

int32_t MtpGenerator::DraftTwo(OrtValue* hidden, int32_t tok0, int32_t tok1) {
  // Populate the [1,2,H] hidden buffer: row 0 = hidden@position L (pairs with tok0), row 1 =
  // hidden@position L+1 (pairs with tok1). `hidden` is the main model's [1,S,H] verify output.
  auto src = ByteWrapTensor(*main_model_.p_device_, *hidden);
  const size_t row_bytes = hidden_slice_->GetByteSpan().size();  // bytes of one [1,1,H] row
  auto dst = hidden_slice2_->GetByteSpan();
  dst.subspan(0, row_bytes).CopyFrom(src.subspan(0, row_bytes));            // row 0 <- hidden@0
  dst.subspan(row_bytes, row_bytes).CopyFrom(src.subspan(row_bytes, row_bytes));  // row 1 <- hidden@1

  // One 2-token MTP forward: feeds tok0 (KV-advance) and tok1 (the next committed token); the
  // last-position logits give the draft for the token after tok1.
  mtp_->SetHiddenStates(hidden_slice2_);
  std::array<int32_t, 2> toks{tok0, tok1};
  mtp_->AppendTokens(cpu_span<const int32_t>(toks));
  auto logits_span = mtp_->GetLogits();              // fp32, last token, [1, V]
  int32_t draft = 0;
  if (mtp_model_.p_device_->ArgMax(logits_span.Span().data(), Ort::TypeToTensorType<float>, 1, vocab_size_, &draft))
    return draft;
  auto logits = logits_span.CopyDeviceToCpu();        // host fallback
  return ArgmaxRow(logits.data(), vocab_size_);
}

void MtpGenerator::AppendTokens(cpu_span<const int32_t> input_ids) {
  main_->AppendTokens(input_ids);
  length_ = input_ids.size();
  for (auto t : input_ids) sequence_.push_back(t);

  OrtValue* hidden = main_->state_->GetOutput(main_model_.config_->model.decoder.outputs.hidden_states.c_str());
  const int last = static_cast<int>(input_ids.size()) - 1;
  ExtractHiddenPosition(hidden, last);             // h for the token we are about to predict
  ArgmaxMainRows(last, 1, &next_token_);           // token predicted for position length_
  has_pending_draft_ = false;
  primed_ = true;
}

void MtpGenerator::GenerateNextToken() {
  if (!primed_) throw std::runtime_error("MtpGenerator: AppendTokens must be called before GenerateNextToken");
  if (done_) return;

  // Commit the token predicted for position length_.
  const int32_t t = next_token_;
  sequence_.push_back(t);
  if (contains(main_model_.config_->model.eos_token_id, t) || sequence_.size() >= static_cast<size_t>(max_length_)) {
    done_ = true;
    return;
  }

  if (num_speculative_tokens_ == 1)
    GenerateStepSingle(t);
  else
    GenerateStepMulti(t);
}

void MtpGenerator::GenerateStepSingle(int32_t t) {
  // 1. Draft the next token for t. After an accepted step the draft was already computed ahead
  //    (fused into that step's KV-advance as one 2-token MTP forward), so reuse it; otherwise the
  //    MTP head is at the right point and we issue a fresh single-token draft.
  int32_t d;
  if (has_pending_draft_) {
    d = pending_draft_;
    has_pending_draft_ = false;
  } else {
    d = DraftNextToken(nullptr, t);  // hidden_slice_ holds h paired with t
  }

  // 2. Snapshot the recurrent state at length L, then verify [t, d] in a single main forward.
  main_->SnapshotState();
  std::array<int32_t, 2> verify{t, d};
  main_->AppendTokens(cpu_span<const int32_t>(verify));
  ++forwards_;

  // Argmax both verify rows on-device in one launch: row 0 = main's real token after t,
  // row 1 = the free prediction harvested when the draft is accepted.
  int32_t verify_argmax[2];
  ArgmaxMainRows(0, 2, verify_argmax);
  const int32_t m = verify_argmax[0];
  ++trials_;

  if (d == m) {
    // 2a. Accept: t and d are both correct. Commit d and harvest the free prediction at row 1.
    ++accepts_;
    sequence_.push_back(d);
    if (sequence_.size() >= static_cast<size_t>(max_length_)) {
      done_ = true;
      return;
    }
    OrtValue* hidden = main_->state_->GetOutput(main_model_.config_->model.decoder.outputs.hidden_states.c_str());
    // Next token to commit is argmax(logits@L+1) (harvested above).
    next_token_ = verify_argmax[1];
    // Fuse the post-accept KV-advance (hidden@L, d) and the next step's draft (hidden@L+1,
    // next_token_) into ONE 2-token MTP forward, and stash the resulting draft for the next step.
    pending_draft_ = DraftTwo(hidden, d, next_token_);
    has_pending_draft_ = true;
    length_ += 2;
  } else {
    // 2b. Reject: roll back the speculative forward (restore recurrent state + crop KV to L),
    //     then re-run the single correct token t. The pipelined draft (if any) is invalid.
    has_pending_draft_ = false;
    main_->RewindToLength(length_);
    std::array<int32_t, 1> rerun{t};
    main_->AppendTokens(cpu_span<const int32_t>(rerun));
    ++forwards_;
    OrtValue* hidden = main_->state_->GetOutput(main_model_.config_->model.decoder.outputs.hidden_states.c_str());
    ArgmaxMainRows(0, 1, &next_token_);
    ExtractHiddenPosition(hidden, 0);
    length_ += 1;
  }
}

void MtpGenerator::GenerateStepMulti(int32_t t) {
  const int N = std::min(num_speculative_tokens_, static_cast<int>(max_length_ - length_ - 1));
  const std::string& hs_name = main_model_.config_->model.decoder.outputs.hidden_states;

  // --- Draft phase: chain the single MTP module N times. ---
  // Step 0 feeds the main model's hidden (hidden_slice_ holds h paired with t) and appends the
  // committed token t to the head KV. Steps 1..N-1 feed the head's OWN post-norm hidden
  // (head_out_hidden_, captured by DraftHeadStep) + the previous draft -- speculative appends.
  const size_t head_start = head_len_;
  mtp_->SetHiddenStates(hidden_slice_);
  drafts_[0] = DraftHeadStep(t);
  for (int k = 1; k < N; ++k) {
    mtp_->SetHiddenStates(head_out_hidden_);
    drafts_[k] = DraftHeadStep(drafts_[k - 1]);
  }

  // --- Verify [t, d0..d_{N-1}] in a single batched main forward (the whole point of MTP: one
  //     forward validates N+1 tokens). The batched (M=N+1) forward is numerically ~equal but not
  //     bit-identical to single-token decode (different GEMM tiling for M=1 vs M>1, plus XQA-vs-
  //     Flash attention), so a greedy argmax can occasionally differ on near-ties. This is the same
  //     tradeoff the N=1 verify already makes; the reject path below re-runs decode-consistently to
  //     bound divergence from plain greedy. ---
  main_->SnapshotState();
  verify_tokens_[0] = t;
  for (int k = 0; k < N; ++k) verify_tokens_[k + 1] = drafts_[k];
  main_->AppendTokens(cpu_span<const int32_t>(verify_tokens_.data(), N + 1));
  ++forwards_;
  ArgmaxMainRows(0, N + 1, verify_argmax_.data());  // main's real token after each verify position
  OrtValue* vhidden = main_->state_->GetOutput(hs_name.c_str());

  // --- Longest accepted prefix (greedy match against the main model). ---
  int a = 0;
  while (a < N && drafts_[a] == verify_argmax_[a]) ++a;
  trials_ += (a < N) ? static_cast<size_t>(a + 1) : static_cast<size_t>(N);  // conditional trials
  accepts_ += static_cast<size_t>(a);

  // Commit the a accepted drafts (t was already committed by the caller). Stop at eos/max_length.
  for (int k = 0; k < a; ++k) {
    sequence_.push_back(drafts_[k]);
    if (contains(main_model_.config_->model.eos_token_id, drafts_[k]) ||
        sequence_.size() >= static_cast<size_t>(max_length_)) {
      done_ = true;
      return;  // generation finished; leftover main/head KV is irrelevant
    }
  }

  // --- Roll the MTP head KV back to the committed tokens: keep t (fed with its main hidden), drop
  //     the N-1 speculative drafts, then re-materialize the a accepted drafts with the main model's
  //     hidden states. Extract the head-refeed hiddens from the verify output BEFORE any main
  //     rewind overwrites the hidden buffer. ---
  mtp_->RewindToLength(head_start + 1);
  head_len_ = head_start + 1;
  for (int k = 0; k < a; ++k) {
    CopyHiddenRow(vhidden, k, *refeed_hidden_);  // main hidden that predicted drafts_[k]
    mtp_->SetHiddenStates(refeed_hidden_);
    DraftHeadStep(drafts_[k], /*need_draft=*/false);  // re-append with the main hidden (no argmax)
  }

  if (a == N) {
    // All drafts accepted: the batched verify already committed [t, d0..d_{N-1}] correctly, so the
    // main KV / recurrent state is exactly at L + (N+1). The bonus token is main's prediction at the
    // last verify row (mirrors the N=1 accept path, which likewise commits a batched-forward token).
    next_token_ = verify_argmax_[a];
    CopyHiddenRow(vhidden, a, *hidden_slice_);
    length_ += static_cast<size_t>(N) + 1;
  } else if (main_->CanCropRecurrentState() && a >= 1) {
    // Partial accept (a>=1), LOSSLESS CROP fast-path (model exported with emit_recurrent_state_all).
    // The batched verify's row a is an EARLY row of a wide (M=N+1) forward, whose argmax is NOT
    // decode-consistent (only the LAST row of a forward matches a 1-token decode; §13.1). So we
    // cannot take the bonus straight from the verify. Instead: crop the KV cache + recurrent state
    // to L+a (state AFTER verify tokens 0..a-1 == present_state_all[:, a-1]) -- avoiding the wide
    // (a+1)-token replay -- then M=1-decode the last committed token (d_{a-1}, at position L+a). Its
    // row-0 logits ARE decode-consistent, giving a lossless bonus. This replaces the §13.5 wide
    // replay (M=a+1) with a cheap M=1 forward.
    // NOTE (§14.3): this is NOT actually lossless in practice -- the cropped state is itself derived
    // from the wide batched verify and differs from sequential decode (~0.25 fp16), so greedy
    // near-ties still flip. Kept for history/reference; the fallback replay below is the lossless path.
    main_->CropToAccepted(length_ + static_cast<size_t>(a), static_cast<size_t>(a) - 1);
    std::array<int32_t, 1> last{drafts_[a - 1]};  // committed token at position L+a
    main_->AppendTokens(cpu_span<const int32_t>(last));
    ++forwards_;
    OrtValue* rhidden = main_->state_->GetOutput(hs_name.c_str());
    ArgmaxMainRows(0, 1, &next_token_);         // decode-consistent bonus (M=1 last row)
    CopyHiddenRow(rhidden, 0, *hidden_slice_);  // hidden paired with the bonus token
    length_ += static_cast<size_t>(a) + 1;
  } else {
    // Rejection at position a: the batched verify over-appended N-a wrong tokens and cannot be
    // partially cropped (the linear-attention recurrent state has no per-token rollback). Restore
    // the recurrent snapshot at L and re-run only the committed prefix [t, d0..d_{a-1}]. Reading the
    // bonus + its hidden from THIS forward keeps the carried state consistent with the committed
    // sequence (the a==0 case degenerates to the exact N=1 single-token decode re-run).
    main_->RewindToLength(length_);
    verify_tokens_[0] = t;
    for (int k = 0; k < a; ++k) verify_tokens_[k + 1] = drafts_[k];
    main_->AppendTokens(cpu_span<const int32_t>(verify_tokens_.data(), a + 1));
    ++forwards_;
    OrtValue* rhidden = main_->state_->GetOutput(hs_name.c_str());
    ArgmaxMainRows(a, 1, &next_token_);        // main's token after the committed prefix
    CopyHiddenRow(rhidden, a, *hidden_slice_);  // hidden paired with the bonus token
    length_ += static_cast<size_t>(a) + 1;
  }
}

bool MtpGenerator::IsDone() const {
  return done_;
}

}  // namespace Generators
