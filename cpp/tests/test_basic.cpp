#include "turboquant/lloyd_max.hpp"
#include "turboquant/turboquant.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

static int fail(const char* msg) {
  std::cerr << "FAIL: " << msg << "\n";
  return 1;
}

int main() {
  using namespace turboquant;

  const auto cb = build_lloyd_max_codebook(128, 2);
  if (cb.num_levels != 4) {
    return fail("num_levels");
  }
  if (cb.centroids.size() != 4u || cb.boundaries.size() != 3u) {
    return fail("sizes");
  }

  TurboQuantKeyCompressor kc(16, 3, 12345u, 99999u);
  if (kc.head_dim() != 16 || kc.mse_bits() != 2) {
    return fail("key compressor config");
  }

  RowMajorMat ortho_check = kc.pi() * kc.pi().transpose();
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      const float want = (i == j) ? 1.f : 0.f;
      if (std::abs(ortho_check(i, j) - want) > 2e-3f) {
        return fail("Pi not orthogonal");
      }
    }
  }

  std::vector<float> x(static_cast<std::size_t>(kc.head_dim()));
  for (int i = 0; i < kc.head_dim(); ++i) {
    x[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
  }

  TurboQuantKeyCompressor::CompressedKey ck;
  kc.compress_row(x.data(), &ck);
  if (ck.k_mse.size() != static_cast<std::size_t>(kc.head_dim()) ||
      ck.signs.size() != static_cast<std::size_t>(kc.head_dim())) {
    return fail("compressed shape");
  }
  if (!(ck.residual_norm >= 0.f)) {
    return fail("residual norm");
  }

  std::vector<float> q(static_cast<std::size_t>(kc.head_dim()), 0.f);
  q[0] = 1.f;
  float score = 0.f;
  kc.asymmetric_attention_scores(q.data(), 1, &ck, 1, &score);
  if (!std::isfinite(score)) {
    return fail("score finite");
  }

  TurboQuantMSECompressor vc(16, 3, 4242u);
  TurboQuantMSECompressor::CompressedValue cv;
  vc.compress_row(x.data(), &cv);
  std::vector<float> x_back(static_cast<std::size_t>(vc.head_dim()));
  vc.decompress_row(cv, x_back.data());
  for (int i = 0; i < vc.head_dim(); ++i) {
    const float v = x_back[static_cast<std::size_t>(i)];
    if (!std::isfinite(v)) {
      return fail("MSE decompress non-finite");
    }
  }

  std::cout << "OK turboquant_cpp tests\n";
  return 0;
}
