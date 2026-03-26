#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "turboquant/turboquant.hpp"

namespace py = pybind11;

namespace {

void expect_contig_2d_f32(const py::buffer_info& r, int expect_cols, const char* name) {
  if (r.ndim != 2) {
    throw std::runtime_error(std::string(name) + ": expected 2-D array");
  }
  if (r.itemsize != static_cast<ssize_t>(sizeof(float))) {
    throw std::runtime_error(std::string(name) + ": expected float32");
  }
  if (r.shape[1] != expect_cols) {
    throw std::runtime_error(std::string(name) + ": wrong trailing dimension");
  }
}

}  // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = R"pbdoc(
    TurboQuant native (C++/Eigen): fast KV key compression and asymmetric attention scores.
    RNG differs from PyTorch for the same integer seeds.
  )pbdoc";

  py::class_<turboquant::TurboQuantKeyCompressor>(m, "KeyCompressor")
      .def(py::init<int, int, std::uint64_t, std::uint64_t>(), py::arg("head_dim"),
           py::arg("bits"), py::arg("seed_pi") = 42ull, py::arg("seed_s") = 10042ull)
      .def_property_readonly("head_dim", &turboquant::TurboQuantKeyCompressor::head_dim)
      .def_property_readonly("bits", &turboquant::TurboQuantKeyCompressor::bits)
      .def_property_readonly("mse_bits", &turboquant::TurboQuantKeyCompressor::mse_bits)
      .def(
          "compress",
          [](const turboquant::TurboQuantKeyCompressor& self,
             py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            auto r = arr.request();
            expect_contig_2d_f32(r, self.head_dim(), "compress");
            const int n = static_cast<int>(r.shape[0]);
            const int d = self.head_dim();
            std::vector<turboquant::TurboQuantKeyCompressor::CompressedKey> keys(
                static_cast<std::size_t>(n));
            self.compress_batch(static_cast<const float*>(r.ptr), n, &keys);

            py::array_t<float> k_mse({n, d});
            py::array_t<std::int8_t> signs({n, d});
            py::array_t<float> rnorm({n});
            auto pk = k_mse.mutable_unchecked<2>();
            auto ps = signs.mutable_unchecked<2>();
            auto pr = rnorm.mutable_unchecked<1>();
            for (int i = 0; i < n; ++i) {
              pr(i) = keys[static_cast<std::size_t>(i)].residual_norm;
              for (int j = 0; j < d; ++j) {
                pk(i, j) = keys[static_cast<std::size_t>(i)]
                               .k_mse[static_cast<std::size_t>(j)];
                ps(i, j) = keys[static_cast<std::size_t>(i)]
                               .signs[static_cast<std::size_t>(j)];
              }
            }
            py::dict out;
            out["k_mse"] = k_mse;
            out["signs"] = signs;
            out["residual_norm"] = rnorm;
            return out;
          },
          py::arg("x"),
          R"pbdoc(
          Compress keys, shape (n, head_dim), float32 C-contiguous.
          Returns dict: k_mse (float32), signs (int8), residual_norm (float32).
        )pbdoc")
      .def(
          "attention_scores",
          [](const turboquant::TurboQuantKeyCompressor& self,
             py::array_t<float, py::array::c_style | py::array::forcecast> queries,
             py::array_t<float, py::array::c_style | py::array::forcecast> k_mse,
             py::array_t<std::int8_t, py::array::c_style> signs,
             py::array_t<float, py::array::c_style | py::array::forcecast> rnorm) {
            auto rq = queries.request();
            auto rk = k_mse.request();
            auto rs = signs.request();
            auto rr = rnorm.request();
            expect_contig_2d_f32(rq, self.head_dim(), "queries");
            expect_contig_2d_f32(rk, self.head_dim(), "k_mse");
            if (rs.ndim != 2 || rs.shape[1] != self.head_dim()) {
              throw std::runtime_error("signs: expected (nk, head_dim) int8");
            }
            if (rr.ndim != 1) {
              throw std::runtime_error("residual_norm: expected 1-D");
            }
            const int nq = static_cast<int>(rq.shape[0]);
            const int nk = static_cast<int>(rk.shape[0]);
            if (rs.shape[0] != nk || rr.shape[0] != nk) {
              throw std::runtime_error("nk mismatch between k_mse, signs, residual_norm");
            }
            py::array_t<float> scores({nq, nk});
            auto rsq = scores.request();
            self.asymmetric_attention_scores_packed(
                static_cast<const float*>(rq.ptr), nq,
                static_cast<const float*>(rk.ptr),
                static_cast<const std::int8_t*>(rs.ptr),
                static_cast<const float*>(rr.ptr), nk,
                static_cast<float*>(rsq.ptr));
            return scores;
          },
          py::arg("queries"), py::arg("k_mse"), py::arg("signs"), py::arg("residual_norm"),
          R"pbdoc(
          queries: (nq, d) float32; k_mse: (nk, d); signs: (nk, d) int8; residual_norm: (nk,).
          Returns scores (nq, nk) float32.
        )pbdoc");

  py::class_<turboquant::TurboQuantMSECompressor>(m, "ValueCompressor")
      .def(py::init<int, int, std::uint64_t>(), py::arg("head_dim"), py::arg("bits"),
           py::arg("seed_pi") = 42ull)
      .def_property_readonly("head_dim", &turboquant::TurboQuantMSECompressor::head_dim)
      .def(
          "compress",
          [](const turboquant::TurboQuantMSECompressor& self,
             py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            auto r = arr.request();
            expect_contig_2d_f32(r, self.head_dim(), "compress");
            const int n = static_cast<int>(r.shape[0]);
            const int d = self.head_dim();
            py::array_t<std::uint8_t> indices({n, d});
            py::array_t<float> norms({n});
            auto pi = indices.mutable_unchecked<2>();
            auto pn = norms.mutable_unchecked<1>();
            for (int i = 0; i < n; ++i) {
              turboquant::TurboQuantMSECompressor::CompressedValue cv;
              self.compress_row(static_cast<const float*>(r.ptr) +
                                    static_cast<std::ptrdiff_t>(i) * d,
                                &cv);
              pn(i) = cv.vec_norm;
              for (int j = 0; j < d; ++j) {
                pi(i, j) = cv.indices[static_cast<std::size_t>(j)];
              }
            }
            py::dict out;
            out["indices"] = indices;
            out["vec_norm"] = norms;
            return out;
          },
          py::arg("x"))
      .def(
          "decompress",
          [](const turboquant::TurboQuantMSECompressor& self,
             py::array_t<std::uint8_t, py::array::c_style> indices,
             py::array_t<float, py::array::c_style | py::array::forcecast> vec_norm) {
            auto ri = indices.request();
            auto rn = vec_norm.request();
            if (ri.ndim != 2 || ri.shape[1] != self.head_dim()) {
              throw std::runtime_error("indices: expected (n, head_dim) uint8");
            }
            if (rn.ndim != 1 || rn.shape[0] != ri.shape[0]) {
              throw std::runtime_error("vec_norm length must match n");
            }
            const int n = static_cast<int>(ri.shape[0]);
            const int d = self.head_dim();
            py::array_t<float> out({n, d});
            auto ro = out.request();
            for (int i = 0; i < n; ++i) {
              turboquant::TurboQuantMSECompressor::CompressedValue cv;
              cv.vec_norm = static_cast<const float*>(rn.ptr)[i];
              cv.indices.resize(static_cast<std::size_t>(d));
              for (int j = 0; j < d; ++j) {
                cv.indices[static_cast<std::size_t>(j)] =
                    static_cast<const std::uint8_t*>(ri.ptr)[i * d + j];
              }
              self.decompress_row(cv, static_cast<float*>(ro.ptr) +
                                           static_cast<std::ptrdiff_t>(i) * d);
            }
            return out;
          },
          py::arg("indices"), py::arg("vec_norm"));

  m.def(
      "lloyd_max_centroids",
      [](int d, int bits) {
        turboquant::LloydMaxCodebook cb = turboquant::build_lloyd_max_codebook(d, bits);
        py::array_t<float> c(static_cast<py::ssize_t>(cb.centroids.size()));
        std::memcpy(c.mutable_data(), cb.centroids.data(),
                    cb.centroids.size() * sizeof(float));
        py::array_t<float> b(static_cast<py::ssize_t>(cb.boundaries.size()));
        std::memcpy(b.mutable_data(), cb.boundaries.data(),
                    cb.boundaries.size() * sizeof(float));
        py::dict out;
        out["centroids"] = c;
        out["boundaries"] = b;
        return out;
      },
      py::arg("d"), py::arg("bits"),
      R"pbdoc(Gaussian-approx Lloyd–Max codebook (same closed form as C++ runtime).)pbdoc");
}
