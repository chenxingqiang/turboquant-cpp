#pragma once

#include <cstdint>
#include <vector>

namespace turboquant {

/// Lloyd–Max centroids for coordinate distribution ~ N(0, 1/d) (Gaussian
/// approximation used in `compressors.py` / default Python path).
struct LloydMaxCodebook {
  int dim_d = 0;
  int bits = 0;
  int num_levels = 0;
  std::vector<float> centroids;
  std::vector<float> boundaries;
};

/// Solve Lloyd–Max for the Gaussian approximation (no SciPy; truncated-normal
/// closed form). `centroids` sorted, `boundaries` length `2^bits - 1`.
LloydMaxCodebook build_lloyd_max_codebook(int d, int bits, int max_iter = 200,
                                          float tol = 1e-10f);

/// Nearest-centroid index for scalar y given sorted centroids (linear scan;
/// small `num_levels` ≤ 16 typical).
std::uint8_t nearest_centroid_index(float y, const float* centroids, int num_levels);

/// Same as above using precomputed midpoints between centroids (faster).
std::uint8_t quantize_scalar_with_boundaries(float y, const float* boundaries,
                                             int num_boundaries);

}  // namespace turboquant
