#pragma once

#include "turboquant/lloyd_max.hpp"

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace turboquant {

using RowMajorMat =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajorVec = Eigen::RowVectorXf;

/// High-performance TurboQuant key compressor matching `TurboQuantCompressorV2`
/// normalization: unit-norm direction + scale recovery, Lloyd–Max on rotated
/// coordinates, QJL signs on residual in original space.
class TurboQuantKeyCompressor {
 public:
  /// `bits`: total bits per coord budget (MSE uses max(bits-1,1), rest is QJL).
  TurboQuantKeyCompressor(int head_dim, int bits, std::uint64_t seed_pi,
                          std::uint64_t seed_s);

  int head_dim() const { return d_; }
  int bits() const { return bits_; }
  int mse_bits() const { return mse_bits_; }

  const RowMajorMat& pi() const { return pi_; }
  const RowMajorMat& s() const { return s_; }
  const std::vector<float>& centroids() const { return codebook_.centroids; }

  /// Per-vector compressed payload (variable length = 2*d floats + 1 float norm
  /// + packed signs; see `compress_row` for layout).
  struct CompressedKey {
    std::vector<float> k_mse;       // d
    std::vector<std::int8_t> signs; // d, values in {-1, +1}
    float residual_norm = 0.f;
  };

  /// One row `x` length d (not necessarily unit norm). Thread-safe if matrices
  /// are fixed (read-only).
  void compress_row(const float* x, CompressedKey* out) const;

  /// Batch compress: `x` row-major (n, d), consecutive rows.
  void compress_batch(const float* x, int n, std::vector<CompressedKey>* out) const;

  /// Asymmetric inner products: `queries` (nq, d), compressed keys length nk.
  /// Result row-major (nq, nk): scores[i*nk + j] = estimate <q_i, k_j>.
  void asymmetric_attention_scores(const float* queries, int nq,
                                   const CompressedKey* keys, int nk,
                                   float* scores_out) const;

  /// Same as above but contiguous packed keys (faster, less allocation).
  void asymmetric_attention_scores_packed(
      const float* queries, int nq, const float* k_mse, const std::int8_t* signs,
      const float* residual_norms, int nk, float* scores_out) const;

 private:
  int d_ = 0;
  int bits_ = 0;
  int mse_bits_ = 0;
  int n_levels_ = 0;
  LloydMaxCodebook codebook_;
  RowMajorMat pi_;
  RowMajorMat s_;
};

/// MSE-only value path (rotation + Lloyd–Max on normalized vectors).
class TurboQuantMSECompressor {
 public:
  TurboQuantMSECompressor(int head_dim, int bits, std::uint64_t seed_pi);

  int head_dim() const { return d_; }

  struct CompressedValue {
    std::vector<std::uint8_t> indices; // d
    float vec_norm = 0.f;
  };

  void compress_row(const float* x, CompressedValue* out) const;
  void decompress_row(const CompressedValue& in, float* x_out) const;

 private:
  int d_ = 0;
  int bits_ = 0;
  int n_levels_ = 0;
  LloydMaxCodebook codebook_;
  RowMajorMat pi_;
};

/// Fill `g` with i.i.d. N(0,1) using `std::mt19937_64` (not bit-identical to PyTorch).
void gaussian_fill(RowMajorMat* g, std::uint64_t seed);

/// Random orthogonal matrix (Haar via Gaussian QR + R-diagonal sign fix).
void random_orthogonal(RowMajorMat* q_out, std::uint64_t seed);

}  // namespace turboquant
