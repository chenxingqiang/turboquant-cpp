#include "turboquant/lloyd_max.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace turboquant {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kSqrt2Pi = 2.5066282746310002f;  // sqrt(2*pi)

inline float standard_normal_pdf(float z) {
  return std::exp(-0.5f * z * z) / kSqrt2Pi;
}

inline float standard_normal_cdf(float z) {
  constexpr float kInvSqrt2 = 0.70710678118654752440f;
  return 0.5f * (1.0f + std::erf(z * kInvSqrt2));  // Phi(z) for standard normal
}

/// E[X | a < X < b] for X ~ N(0, sigma^2), sigma > 0.
float truncated_normal_mean(float a, float b, float sigma) {
  const float alpha = a / sigma;
  const float beta = b / sigma;
  const float phi_a = standard_normal_pdf(alpha);
  const float phi_b = standard_normal_pdf(beta);
  const float Phi_a = standard_normal_cdf(alpha);
  const float Phi_b = standard_normal_cdf(beta);
  const float denom = Phi_b - Phi_a;
  if (denom <= 1e-20f) {
    return 0.5f * (a + b);
  }
  return sigma * (phi_a - phi_b) / denom;
}

}  // namespace

LloydMaxCodebook build_lloyd_max_codebook(int d, int bits, int max_iter,
                                          float tol) {
  LloydMaxCodebook out;
  out.dim_d = d;
  out.bits = bits;
  out.num_levels = 1 << bits;
  const int n = out.num_levels;
  const float sigma = 1.0f / std::sqrt(static_cast<float>(d));
  const float lo = -3.5f * sigma;
  const float hi = 3.5f * sigma;

  std::vector<float> centroids(n);
  for (int i = 0; i < n; ++i) {
    centroids[static_cast<std::size_t>(i)] =
        lo + (hi - lo) * (static_cast<float>(i) + 0.5f) / static_cast<float>(n);
  }

  std::vector<float> boundaries(static_cast<std::size_t>(std::max(0, n - 1)));

  for (int it = 0; it < max_iter; ++it) {
    for (int i = 0; i < n - 1; ++i) {
      boundaries[static_cast<std::size_t>(i)] =
          0.5f * (centroids[static_cast<std::size_t>(i)] +
                  centroids[static_cast<std::size_t>(i + 1)]);
    }

    const float edge_left = lo * 3.0f;
    const float edge_right = hi * 3.0f;
    std::vector<float> new_c(static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i) {
      const float a = (i == 0) ? edge_left : boundaries[static_cast<std::size_t>(i - 1)];
      const float b = (i == n - 1) ? edge_right : boundaries[static_cast<std::size_t>(i)];
      new_c[static_cast<std::size_t>(i)] = truncated_normal_mean(a, b, sigma);
    }

    float max_shift = 0.0f;
    for (int i = 0; i < n; ++i) {
      max_shift = std::max(max_shift,
                           std::abs(new_c[static_cast<std::size_t>(i)] -
                                    centroids[static_cast<std::size_t>(i)]));
    }
    centroids.swap(new_c);
    if (max_shift < tol) {
      break;
    }
  }

  for (int i = 0; i < n - 1; ++i) {
    boundaries[static_cast<std::size_t>(i)] =
        0.5f * (centroids[static_cast<std::size_t>(i)] +
                centroids[static_cast<std::size_t>(i + 1)]);
  }

  out.centroids = std::move(centroids);
  out.boundaries = std::move(boundaries);
  return out;
}

std::uint8_t nearest_centroid_index(float y, const float* centroids,
                                    int num_levels) {
  int best = 0;
  float best_d = std::abs(y - centroids[0]);
  for (int i = 1; i < num_levels; ++i) {
    const float di = std::abs(y - centroids[i]);
    if (di < best_d) {
      best_d = di;
      best = i;
    }
  }
  return static_cast<std::uint8_t>(best);
}

std::uint8_t quantize_scalar_with_boundaries(float y, const float* boundaries,
                                             int num_boundaries) {
  // boundaries size = num_levels - 1, cell i if boundaries[i-1] <= y < boundaries[i]
  int lo_idx = 0;
  int hi_idx = num_boundaries;  // num_levels - 1
  while (lo_idx < hi_idx) {
    const int mid = (lo_idx + hi_idx) / 2;
    if (y < boundaries[mid]) {
      hi_idx = mid;
    } else {
      lo_idx = mid + 1;
    }
  }
  return static_cast<std::uint8_t>(lo_idx);
}

}  // namespace turboquant
