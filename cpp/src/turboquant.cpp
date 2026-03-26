#include "turboquant/turboquant.hpp"

#include <cmath>
#include <random>

#if TURBOQUANT_USE_OPENMP
#include <omp.h>
#endif

#ifndef TURBOQUANT_USE_OPENMP
#define TURBOQUANT_USE_OPENMP 0
#endif

namespace turboquant {
namespace {

constexpr float kPi = 3.14159265358979323846f;

inline float norm_l2(const float* x, int d) {
  float s = 0.f;
  for (int i = 0; i < d; ++i) {
    s += x[i] * x[i];
  }
  return std::sqrt(s);
}

}  // namespace

void gaussian_fill(RowMajorMat* g, std::uint64_t seed) {
  std::mt19937_64 gen(seed);
  std::normal_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < g->rows(); ++i) {
    for (int j = 0; j < g->cols(); ++j) {
      (*g)(i, j) = dist(gen);
    }
  }
}

void random_orthogonal(RowMajorMat* q_out, std::uint64_t seed) {
  const int d = static_cast<int>(q_out->rows());
  Eigen::MatrixXf g(d, d);
  {
    std::mt19937_64 gen(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < d; ++i) {
      for (int j = 0; j < d; ++j) {
        g(i, j) = dist(gen);
      }
    }
  }
  Eigen::HouseholderQR<Eigen::MatrixXf> qr(g);
  Eigen::MatrixXf q = qr.householderQ() * Eigen::MatrixXf::Identity(d, d);
  Eigen::MatrixXf r_upper = qr.matrixQR().triangularView<Eigen::Upper>();
  for (int j = 0; j < d; ++j) {
    float s = (r_upper(j, j) >= 0.f) ? 1.f : -1.f;
    if (r_upper(j, j) == 0.f) {
      s = 1.f;
    }
    q.col(j) *= s;
  }
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < d; ++j) {
      (*q_out)(i, j) = q(i, j);
    }
  }
}

TurboQuantKeyCompressor::TurboQuantKeyCompressor(int head_dim, int bits,
                                                 std::uint64_t seed_pi,
                                                 std::uint64_t seed_s)
    : d_(head_dim),
      bits_(bits),
      mse_bits_(std::max(bits - 1, 1)),
      n_levels_(1 << mse_bits_),
      codebook_(build_lloyd_max_codebook(head_dim, mse_bits_)),
      pi_(head_dim, head_dim),
      s_(head_dim, head_dim) {
  random_orthogonal(&pi_, seed_pi);
  gaussian_fill(&s_, seed_s);
}

void TurboQuantKeyCompressor::compress_row(const float* x,
                                           CompressedKey* out) const {
  out->k_mse.resize(static_cast<std::size_t>(d_));
  out->signs.resize(static_cast<std::size_t>(d_));

  const float norm = norm_l2(x, d_);
  const float inv_norm = (norm > 1e-8f) ? (1.f / norm) : 0.f;

  Eigen::RowVectorXf xn(d_);
  for (int i = 0; i < d_; ++i) {
    xn(i) = x[i] * inv_norm;
  }

  const Eigen::RowVectorXf rotated = xn * pi_.transpose();
  const float* ctr = codebook_.centroids.data();

  Eigen::RowVectorXf recon_rot(d_);
  for (int i = 0; i < d_; ++i) {
    const float y = rotated(i);
    const std::uint8_t idx = nearest_centroid_index(y, ctr, n_levels_);
    recon_rot(i) = ctr[idx];
  }

  const Eigen::RowVectorXf k_mse_row = recon_rot * pi_;
  for (int i = 0; i < d_; ++i) {
    out->k_mse[static_cast<std::size_t>(i)] = k_mse_row(i) * norm;
  }

  float rnorm2 = 0.f;
  for (int i = 0; i < d_; ++i) {
    const float res = x[i] - out->k_mse[static_cast<std::size_t>(i)];
    rnorm2 += res * res;
  }
  out->residual_norm = std::sqrt(rnorm2);

  Eigen::RowVectorXf resv(d_);
  for (int i = 0; i < d_; ++i) {
    resv(i) = x[i] - out->k_mse[static_cast<std::size_t>(i)];
  }
  const Eigen::RowVectorXf proj = resv * s_.transpose();
  for (int i = 0; i < d_; ++i) {
    const float v = proj(i);
    out->signs[static_cast<std::size_t>(i)] =
        (v >= 0.f) ? static_cast<std::int8_t>(1) : static_cast<std::int8_t>(-1);
  }
}

void TurboQuantKeyCompressor::compress_batch(
    const float* x, int n, std::vector<CompressedKey>* out) const {
  out->resize(static_cast<std::size_t>(n));
#if TURBOQUANT_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int r = 0; r < n; ++r) {
    compress_row(x + static_cast<std::ptrdiff_t>(r) * d_,
                  &(*out)[static_cast<std::size_t>(r)]);
  }
}

void TurboQuantKeyCompressor::asymmetric_attention_scores_packed(
    const float* queries, int nq, const float* k_mse, const std::int8_t* signs,
    const float* residual_norms, int nk, float* scores_out) const {
  const int d = d_;
  const int m = d;  // S is (d,d) in this implementation
  const float correction_scale = std::sqrt(kPi / 2.f) / static_cast<float>(m);

  Eigen::Map<const RowMajorMat> Q(queries, nq, d);
  Eigen::Map<const RowMajorMat> K(k_mse, nk, d);

  RowMajorMat term1 = Q * K.transpose();
  RowMajorMat Qp = Q * s_.transpose();

  RowMajorMat Sgnf(nk, d);
  for (int i = 0; i < nk; ++i) {
    for (int j = 0; j < d; ++j) {
      Sgnf(i, j) = static_cast<float>(
          signs[static_cast<std::size_t>(i) * d + j]);
    }
  }
  RowMajorMat term2 = Qp * Sgnf.transpose();

  Eigen::Map<RowMajorMat> out(scores_out, nq, nk);
  for (int i = 0; i < nq; ++i) {
    for (int j = 0; j < nk; ++j) {
      out(i, j) = term1(i, j) +
                  correction_scale * term2(i, j) * residual_norms[j];
    }
  }
}

void TurboQuantKeyCompressor::asymmetric_attention_scores(
    const float* queries, int nq, const CompressedKey* keys, int nk,
    float* scores_out) const {
  std::vector<float> kflat(static_cast<std::size_t>(nk * d_));
  std::vector<std::int8_t> sflat(static_cast<std::size_t>(nk * d_));
  std::vector<float> norms(static_cast<std::size_t>(nk));
  for (int j = 0; j < nk; ++j) {
    norms[static_cast<std::size_t>(j)] = keys[j].residual_norm;
    for (int t = 0; t < d_; ++t) {
      kflat[static_cast<std::size_t>(j * d_ + t)] =
          keys[j].k_mse[static_cast<std::size_t>(t)];
      sflat[static_cast<std::size_t>(j * d_ + t)] =
          keys[j].signs[static_cast<std::size_t>(t)];
    }
  }
  asymmetric_attention_scores_packed(queries, nq, kflat.data(), sflat.data(),
                                     norms.data(), nk, scores_out);
}

TurboQuantMSECompressor::TurboQuantMSECompressor(int head_dim, int bits,
                                                 std::uint64_t seed_pi)
    : d_(head_dim),
      bits_(bits),
      n_levels_(1 << bits_),
      codebook_(build_lloyd_max_codebook(head_dim, bits)),
      pi_(head_dim, head_dim) {
  random_orthogonal(&pi_, seed_pi);
}

void TurboQuantMSECompressor::compress_row(const float* x,
                                           CompressedValue* out) const {
  out->indices.resize(static_cast<std::size_t>(d_));
  const float norm = norm_l2(x, d_);
  const float inv_norm = (norm > 1e-8f) ? (1.f / norm) : 0.f;
  out->vec_norm = norm;

  Eigen::RowVectorXf xn(d_);
  for (int i = 0; i < d_; ++i) {
    xn(i) = x[i] * inv_norm;
  }
  const Eigen::RowVectorXf rotated = xn * pi_.transpose();
  const float* ctr = codebook_.centroids.data();
  for (int i = 0; i < d_; ++i) {
    const float y = rotated(i);
    const std::uint8_t idx = nearest_centroid_index(y, ctr, n_levels_);
    out->indices[static_cast<std::size_t>(i)] = idx;
  }
}

void TurboQuantMSECompressor::decompress_row(const CompressedValue& in,
                                             float* x_out) const {
  const float* ctr = codebook_.centroids.data();
  Eigen::RowVectorXf recon_rot(d_);
  for (int i = 0; i < d_; ++i) {
    recon_rot(i) = ctr[in.indices[static_cast<std::size_t>(i)]];
  }
  const Eigen::RowVectorXf recon = recon_rot * pi_;
  for (int i = 0; i < d_; ++i) {
    x_out[i] = recon(i) * in.vec_norm;
  }
}

}  // namespace turboquant
