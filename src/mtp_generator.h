// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <vector>

namespace Generators {

struct Model;
struct Generator;
struct GeneratorParams;
struct Tensor;

// In-engine Multi-Token-Prediction (MTP) self-speculative decoder for Qwen3.6-style models.
//
// It composes two genai generators on the shared compute stream:
//   * the main decoder (exported with include_hidden_states so it emits a hidden_states output)
//   * the MTP head (mtp.onnx, a single decoder layer that drafts the next-next token)
//
// The main model's last hidden state is handed to the MTP head device-to-device (no host
// round-trip), and the draft is verified against the main model in a single 2-token forward.
// Greedy, batch size 1. The output is identical to plain greedy decoding (lossless), modulo
// floating-point near-ties in the batched verify forward.
struct MtpGenerator {
  MtpGenerator(const Model& main_model, const Model& mtp_model, const GeneratorParams& params);

  // Seed the prompt (runs the main model's prefill).
  void AppendTokens(cpu_span<const int32_t> input_ids);

  // Produce the next token via the draft/verify loop. On an accepted draft this commits two
  // tokens (and harvests a free third prediction from the verify pass); on a rejected draft it
  // commits one and rolls back the speculative forward.
  void GenerateNextToken();

  bool IsDone() const;

  // The full committed token sequence (batch index 0).
  const std::vector<int32_t>& GetSequence() const { return sequence_; }

  // Speculative-decoding statistics.
  size_t Forwards() const { return forwards_; }
  size_t Accepts() const { return accepts_; }
  size_t Trials() const { return trials_; }

 private:
  // Run the MTP head on a single (hidden_state, token) pair. When `need_draft` is true, returns the
  // head's greedy drafted next token; when false, only advances the head's KV cache (skipping the
  // 248K-vocab argmax + its stream sync) and returns 0. The KV-advance-only mode is used after an
  // accepted draft, where the next token comes from the verify pass rather than a fresh draft.
  int32_t DraftNextToken(OrtValue* hidden_last_position, int32_t token, bool need_draft = true);
  // Feed the MTP head two tokens in one forward: (hidden@row0, tok0) then (hidden@row1, tok1),
  // where hidden rows come from `hidden` (a [1,S,H] verify output). Returns the greedy argmax of
  // the last (tok1) position -- the draft for the token after tok1. This fuses the post-accept
  // KV-advance (tok0) and the next step's draft (tok1) into a single 2-token MTP forward.
  int32_t DraftTwo(OrtValue* hidden, int32_t tok0, int32_t tok1);
  // Copy one [1,1,H] position out of a [1,S,H] hidden_states OrtValue into hidden_slice_ (D2D).
  void ExtractHiddenPosition(OrtValue* hidden, int position);
  // Copy one [1,1,H] row out of a [1,S,H] hidden OrtValue (on `main_model_`'s device) into `dst`.
  void CopyHiddenRow(OrtValue* hidden, int position, Tensor& dst);
  // One MTP-head forward on a single token: the head's `hidden_states` input must already be set
  // (via SetHiddenStates) by the caller. Appends `token` to the head KV, captures the head's own
  // post-final-norm output (hidden_states_out, last row) into `head_out_hidden_` for the next
  // chained step, and returns the greedy draft (or 0 if need_draft is false).
  int32_t DraftHeadStep(int32_t token, bool need_draft = true);
  // Single-token (num_speculative_tokens == 1) draft/verify step (the original fast path).
  void GenerateStepSingle(int32_t t);
  // Multi-token (num_speculative_tokens > 1) chained draft/verify step: chains the single MTP
  // module N times (feeding its own hidden back), verifies [t, d0..d_{N-1}] in one main forward,
  // commits the longest accepted prefix + 1 bonus, and rolls the head/main state back losslessly.
  void GenerateStepMulti(int32_t t);
  // Greedy argmax over `num_rows` consecutive vocab rows of the main model's raw logits output
  // ([1,S,V]), starting at `first_row`, writing the token ids to `out`. Uses the device's
  // on-device Top-K kernel when available (no full-logits host copy); falls back to a host argmax.
  void ArgmaxMainRows(int first_row, int num_rows, int32_t* out);

  const Model& main_model_;
  const Model& mtp_model_;

  std::unique_ptr<Generator> main_;  // main decoder generator
  std::unique_ptr<Generator> mtp_;   // MTP head generator (drafts)
  // State retains GeneratorParams via shared_from_this, so the head needs a distinct persistent
  // parameter object. It must use the head's Config and keep graph capture off (head graph replay
  // corrupts chained drafts; the main model alone captures the verify shapes).
  std::shared_ptr<GeneratorParams> mtp_params_;

  std::shared_ptr<Tensor> hidden_slice_;  // reusable [1,1,hidden] device buffer for the handoff
  std::shared_ptr<Tensor> hidden_slice2_;  // reusable [1,2,hidden] buffer for the batched 2-token draft
  std::shared_ptr<Tensor> head_out_hidden_;  // [1,1,hidden] capture of the head's own hidden (chain feedback)
  std::shared_ptr<Tensor> refeed_hidden_;    // [1,1,hidden] scratch for re-feeding accepted drafts (main hidden)
  std::unique_ptr<OrtValue> logits_fp32_;  // reusable fp32 cast of the main model's raw logits

  std::vector<int32_t> sequence_;  // committed tokens (batch 0)
  int hidden_size_{};
  int vocab_size_{};
  int max_length_{};

  // Number of speculative draft tokens per step (N). 1 = the original single-token fast path;
  // >1 chains the single MTP module N times (Qwen3.6 / vLLM-style). Read from the
  // ORT_MTP_NUM_SPECULATIVE_TOKENS env var at construction (default 1).
  int num_speculative_tokens_{1};
  bool decode_consistent_accept_{false};
  // Head KV length invariant (multi-token path): number of committed generated tokens currently
  // in the MTP head's KV cache (each fed once with its main hidden). The draft phase temporarily
  // extends this speculatively, then rolls it back to this value + accepted drafts.
  size_t head_len_{0};
  std::vector<int32_t> drafts_;         // scratch: the N chained draft tokens
  std::vector<int32_t> verify_tokens_;  // scratch: [t, d0..d_{N-1}] for the verify forward
  std::vector<int32_t> verify_argmax_;  // scratch: main argmax of the N+1 verify rows

  // Loop carry state (see the design doc draft/verify invariant):
  int32_t next_token_{};   // token predicted for the current cache length L (not yet committed)
  size_t length_{};        // committed cache length L
  bool primed_{false};     // whether AppendTokens has run the prompt
  bool done_{false};
  // Pipelined draft: on an accepted step the next step's draft is computed ahead (fused into the
  // post-accept KV-advance as one 2-token MTP forward), so the next GenerateNextToken reuses it
  // instead of issuing a separate draft forward.
  int32_t pending_draft_{};
  bool has_pending_draft_{false};

  size_t forwards_{};
  size_t accepts_{};
  size_t trials_{};
};

}  // namespace Generators
