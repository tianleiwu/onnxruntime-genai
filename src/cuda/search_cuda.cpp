// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "interface.h"
#include "search.h"
#include "search_cuda.h"
#include "beam_search_scorer_cuda.cuh"
#include "beam_search_scorer_cuda.h"
#include "beam_search_topk.h"
#include "cuda_topk.h"
#include <queue>
#include <random>
#include <iostream>
#include <vector>
#include <iomanip>

namespace Generators {

void OnCudaError(cudaError_t error) {
  printf("Cuda Error: %s\n", cudaGetErrorString(error));
  assert(false);
  throw std::exception();
}

Search_Cuda::Search_Cuda(const GeneratorParams& params)
    : Search{params} {
  auto batch_beam_size = params.BatchBeamSize();
  sequence_lengths_ = params.p_device->Allocate<int32_t>(batch_beam_size);

  eos_seen_buffer_ = CudaMallocArray<bool>(batch_beam_size, &eos_seen_);
  cudaMemsetAsync(eos_seen_.data(), 0, eos_seen_.size_bytes(), GetStream());

  eos_token_ids_ = params.p_device->Allocate<int32_t>(params.config.model.eos_token_id.size());
  copy(std::span<const int32_t>{params.config.model.eos_token_id}, eos_token_ids_.CpuSpan());
  eos_token_ids_.CopyCpuToDevice();

  done_cpu_ = CudaMallocHostArray<bool>(1);
  *done_cpu_ = false;
}

GreedySearch_Cuda::GreedySearch_Cuda(const GeneratorParams& params)
    : Search_Cuda{params} {
  next_tokens_buffer_ = params.p_device->Allocate<int32_t>(params.search.batch_size);
  next_tokens_buffer_.Zero();
  next_tokens_ = gpu_span<int32_t>(next_tokens_buffer_.Span());

  unsigned long long random_seed = (params_->search.random_seed != -1)
                                       ? params_->search.random_seed
                                       : std::random_device{}();

  // Allocate a single buffer for all sampling data
  size_t sampling_buffer_size = cuda::SamplingData::CalculateTotalSize(params.search.batch_size, params.config.model.vocab_size, GetStream());
  sampling_buffer_ = params.p_device->Allocate<uint8_t>(sampling_buffer_size);

  // Create SamplingData with the externally allocated buffer
  samplingdata_ = std::make_unique<cuda::SamplingData>(random_seed, params.search.batch_size, params.config.model.vocab_size, GetStream(),
                                                       sampling_buffer_.Span().data(), sampling_buffer_size);
}

BeamSearch_Cuda::BeamSearch_Cuda(const GeneratorParams& params)
    : Search_Cuda{params} {
  assert(params_->search.num_beams > 1);  // If 1, use GreedySearch
  auto batch_beam_size = params_->BatchBeamSize();
  beam_scorer_ = std::make_unique<BeamSearchScorer_Cuda>(*params_, eos_token_ids_.Span());

  topk_next_tokens_ = CudaMallocArray<int32_t>(2 * batch_beam_size);
  topk_next_indices_ = CudaMallocArray<int32_t>(2 * batch_beam_size);
  topk_next_scores_ = CudaMallocArray<float>(2 * batch_beam_size);
  softmax_buffer_ = CudaMallocArray<float>(batch_beam_size * params_->config.model.vocab_size);

  const int k_vocab_top_k = 2 * 16; // 2 * params_->search.num_beams;
  topk_tmp_scores_ = CudaMallocArray<float>(batch_beam_size * k_vocab_top_k);
  topk_tmp_tokens_ = CudaMallocArray<int32_t>(batch_beam_size * k_vocab_top_k);

  size_t topk_buffer_size = cuda::TopkData::CalculateTotalSize(batch_beam_size, params.config.model.vocab_size, GetStream());
  topk_buffer_ = params.p_device->Allocate<uint8_t>(topk_buffer_size);

  // Create TopkData with the externally allocated buffer
  topk_data_ = std::make_unique<cuda::TopkData>(batch_beam_size, params.config.model.vocab_size, GetStream(),
                                               topk_buffer_.Span().data(), topk_buffer_size);
}

BeamSearch_Cuda::~BeamSearch_Cuda() = default;

DeviceSpan<float> Search_Cuda::GetLogits() const {
  return next_token_scores_;
}

void Search_Cuda::SetLogits(DeviceSpan<float> logits) {
  next_token_scores_ = logits;
}

DeviceSpan<int32_t> GreedySearch_Cuda::GetNextTokens() {
  return next_tokens_buffer_;
}

DeviceSpan<int32_t> BeamSearch_Cuda::GetNextTokens() {
  return beam_scorer_->GetNextTokens();
}

DeviceSpan<int32_t> BeamSearch_Cuda::GetNextIndices() {
  return beam_scorer_->GetNextIndices();
}

void BeamSearch_Cuda::SelectTop() {
  cuda::DispatchBlockwiseSoftmaxForward<true>(GetStream(), softmax_buffer_.get(), next_token_scores_.Span().data(), params_->config.model.vocab_size,
                                              params_->config.model.vocab_size, params_->config.model.vocab_size, params_->BatchBeamSize());

  auto beam_scores = beam_scorer_->GetNextScores();

  // Add beam score to next token scores. Corresponding python code is like:
  //    next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
  cuda::LaunchAddProbsKernel(softmax_buffer_.get(), beam_scores.Span().data(),
                             params_->search.batch_size, params_->search.num_beams, params_->config.model.vocab_size, GetStream());

  cuda::BeamSearchTopK(topk_data_.get(),
                       softmax_buffer_.get(),
                       params_->search.batch_size,
                       params_->search.num_beams,
                       params_->config.model.vocab_size,
                       topk_tmp_scores_.get(),
                       topk_tmp_tokens_.get(),
                       topk_next_scores_.get(),
                       topk_next_tokens_.get(),
                       topk_next_indices_.get(),
                       GetStream());
      
  CudaCheck() == cudaStreamSynchronize(GetStream());

  const size_t valid_candidate_count = static_cast<size_t>(params_->search.batch_size) * params_->search.num_beams;
  std::span<const float> next_scores{topk_next_scores_.get(), valid_candidate_count};
  std::span<const int32_t> next_tokens{topk_next_tokens_.get(), valid_candidate_count};
  std::span<const int32_t> next_indices{topk_next_indices_.get(), valid_candidate_count};

  beam_scorer_->Process(sequences_, next_scores, next_tokens, next_indices);
  auto next_tokens_device = beam_scorer_->GetNextTokens();
  next_tokens_ = gpu_span<int32_t>(next_tokens_device.Span());
  sequences_.AfterAppendNextTokens(next_tokens_device, params_->BatchBeamSize());
}

void GreedySearch_Cuda::SampleTopKTopP(int k, float p, float temperature) {
  std::span<float> scores = next_token_scores_.Span();
  assert(scores.size() == params_->search.batch_size * params_->config.model.vocab_size);
  cuda::GetSample(samplingdata_.get(), GetStream(), next_tokens_.data(), scores.data(), int(scores.size() / params_->search.batch_size),
                  params_->search.batch_size, k, p, temperature);

  // Check for EOS
  assert(next_tokens_.size() == eos_seen_.size());
  cuda::Launch_CheckForEOSAndPad(next_tokens_.data(), static_cast<int>(next_tokens_.size()), eos_seen_.data(), eos_token_ids_.Span().data(), static_cast<int>(eos_token_ids_.Span().size()), params_->config.model.pad_token_id, done_cpu_.get(), GetStream());

  // Append tokens
  cuda::Launch_AppendNextTokensToSequences(next_tokens_buffer_.Span(), sequences_.GetSequences().Span(), params_->BatchBeamSize(), sequences_.GetSequenceLength(), sequences_.max_length_, GetStream());
  sequences_.AfterAppendNextTokens(next_tokens_buffer_, params_->BatchBeamSize());

  if (sequences_.GetSequenceLength() == params_->search.max_length) {
    if (GetLogItems().enabled && GetLogItems().hit_max_length)
      Log("hit_max_length", "greedy cuda hit");
    *done_cpu_ = true;
  }
}

bool BeamSearch_Cuda::IsDone() const {
  if (beam_scorer_->IsDoneLater())
    return true;

  if (sequences_.GetSequenceLength() == params_->search.max_length) {
    if (GetLogItems().enabled && GetLogItems().hit_max_length)
      Log("hit_max_length", "beam cuda hit");
    return true;
  }
  return false;
}

void BeamSearch_Cuda::Finalize(size_t num_return_sequences) {
  if (finalized_)
    return;
  beam_scorer_->Finalize(sequences_, num_return_sequences);
  finalized_ = true;
}

DeviceSpan<int32_t> BeamSearch_Cuda::GetSequence(size_t index) {
  Finalize(params_->search.num_return_sequences);
  const size_t batch_id = index / params_->search.num_return_sequences;
  const size_t beam_id = index % params_->search.num_return_sequences;
  return beam_scorer_->GetBeamHypothesis(batch_id, beam_id);
}

DeviceSpan<int32_t> BeamSearch_Cuda::GetSequence(size_t batch_id, size_t beam_id) {
  Finalize(params_->search.num_return_sequences);
  return beam_scorer_->GetBeamHypothesis(batch_id, beam_id);
}

std::span<float> Search_Cuda::GetScores(int batch_beam_index) {
  assert(batch_beam_index >= 0 && batch_beam_index < params_->BatchBeamSize());
  return next_token_scores_.subspan(batch_beam_index * params_->config.model.vocab_size, params_->config.model.vocab_size).Span();
}

std::span<float> Search_Cuda::GetScores() {
  return next_token_scores_.Span();
}

// Set user input tokens (batch_beam_size, sequence_length)
void GreedySearch_Cuda::AppendTokens(DeviceSpan<int32_t>& next_tokens) {
  cudaMemsetAsync(eos_seen_.data(), 0, eos_seen_.size_bytes(), GetStream());
  *done_cpu_ = false;

  auto next_tokens_gpu = next_tokens.Span();
  cuda::Launch_AppendNextTokensToSequences(next_tokens_gpu, sequences_.GetSequences().Span(), params_->BatchBeamSize(), sequences_.GetSequenceLength(), sequences_.max_length_, GetStream());
  sequences_.AfterAppendNextTokens(next_tokens, params_->BatchBeamSize());

  if (sequences_.GetSequenceLength() >= params_->search.max_length) {
    if (GetLogItems().enabled && GetLogItems().hit_max_length)
      Log("hit_max_length", "greedy cuda hit");
    *done_cpu_ = true;
    return;
  }

  cudaMemsetAsync(eos_seen_.data(), 0, eos_seen_.size_bytes(), GetStream());
  *done_cpu_ = false;
}

void BeamSearch_Cuda::AppendTokens(DeviceSpan<int32_t>& next_tokens) {
  auto next_tokens_gpu = next_tokens.Span();
  cuda::Launch_ExpandInputSequences(next_tokens_gpu, sequences_.GetNextSequences().Span(), params_->search.batch_size, params_->search.num_beams, sequences_.max_length_, GetStream());
  cuda::Launch_ExpandInputSequences(next_tokens_gpu, sequences_.GetSequences().Span(), params_->search.batch_size, params_->search.num_beams, sequences_.max_length_, GetStream());
  sequences_.AfterAppendNextTokens(next_tokens, params_->search.batch_size);  // next_tokens is batch_size
  cudaStreamSynchronize(GetStream());
}

void GreedySearch_Cuda::RewindTo(size_t index) {
  cudaMemsetAsync(eos_seen_.data(), 0, eos_seen_.size_bytes(), GetStream());
  *done_cpu_ = false;
  if (index > 0)
    cuda::Launch_GetLastTokens(next_tokens_.data(), sequences_.GetSequences().Span().data(), static_cast<int>(params_->BatchBeamSize()), static_cast<int>(index), sequences_.max_length_, GetStream());
  else
    cudaMemsetAsync(next_tokens_.data(), 0, params_->search.batch_size * sizeof(int32_t), GetStream());
  sequences_.RewindTo(index);
}

void Search_Cuda::ApplyMinLength(int min_length) {
  if (sequences_.GetSequenceLength() >= min_length)
    return;

  for (auto eos_token_id : params_->config.model.eos_token_id)
    cuda::LaunchSetScoreProcessor(GetScores().data(), params_->BatchBeamSize(), params_->config.model.vocab_size, eos_token_id, std::numeric_limits<float>::lowest(), GetStream());
}

void Search_Cuda::ApplyRepetitionPenalty(float penalty) {
  if (penalty == 1.0f)
    return;

  cuda::LaunchRepetitionPenaltyProcessor(sequences_.GetSequences().Span().data(),
                                         GetScores().data(), params_->search.batch_size, params_->search.num_beams, params_->config.model.vocab_size,
                                         params_->search.max_length, GetSequenceLength(), penalty, GetStream());
}

}  // namespace Generators

