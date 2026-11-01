// pybind11 module `ikernels`: exposes the C++ kernels to Python/NumPy for comparison with PyTorch.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include "ik/cpu_kernels.hpp"
#ifdef IK_HAVE_CUDA
#include "ik/cuda_kernels.hpp"
#endif

namespace py = pybind11;
using Arr = py::array_t<float, py::array::c_style | py::array::forcecast>;

static void check2d(const Arr& a, const char* name) {
  if (a.ndim() != 2) throw std::invalid_argument(std::string(name) + " must be 2-D");
}

static Arr matmul(Arr A, Arr B, bool blocked) {
  check2d(A, "A"); check2d(B, "B");
  const int M = A.shape(0), K = A.shape(1), N = B.shape(1);
  if (B.shape(0) != K) throw std::invalid_argument("inner dimensions differ");
  Arr C({M, N});
  {
    py::gil_scoped_release nogil;
    (blocked ? ik::cpu::matmul_blocked : ik::cpu::matmul_naive)(A.data(), B.data(), C.mutable_data(), M, N, K);
  }
  return C;
}

static Arr softmax(Arr x, bool fused) {
  check2d(x, "x");
  Arr y({x.shape(0), x.shape(1)});
  {
    py::gil_scoped_release nogil;
    (fused ? ik::cpu::softmax_fused : ik::cpu::softmax_naive)(x.data(), y.mutable_data(), x.shape(0), x.shape(1));
  }
  return y;
}

static Arr layernorm(Arr x, Arr g, Arr b, float eps, bool fused) {
  check2d(x, "x");
  if (g.size() != x.shape(1) || b.size() != x.shape(1)) throw std::invalid_argument("gamma/beta size");
  Arr y({x.shape(0), x.shape(1)});
  {
    py::gil_scoped_release nogil;
    (fused ? ik::cpu::layernorm_fused : ik::cpu::layernorm_naive)(x.data(), g.data(), b.data(), y.mutable_data(),
                                                                   x.shape(0), x.shape(1), eps);
  }
  return y;
}

PYBIND11_MODULE(ikernels, m) {
  m.doc() = "Hand-written C++/CUDA inference kernels";
  m.def("matmul", &matmul, py::arg("A"), py::arg("B"), py::arg("blocked") = true);
  m.def("softmax", &softmax, py::arg("x"), py::arg("fused") = true);
  m.def("layernorm", &layernorm, py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5f,
        py::arg("fused") = true);
#ifdef IK_HAVE_CUDA
  m.attr("has_cuda") = true;
  m.def("cuda_device_available", &ik::cuda::device_available);
  m.def("cuda_bench_matmul", &ik::cuda::bench_matmul, py::arg("variant"), py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("iters") = 20, "CUDA-event ms per launch; variant 0=naive 1=tiled 2=reg-tiled");
  m.def("cuda_bench_softmax", &ik::cuda::bench_softmax, py::arg("rows"), py::arg("cols"), py::arg("iters") = 50);
  m.def("cuda_bench_layernorm", &ik::cuda::bench_layernorm, py::arg("rows"), py::arg("cols"), py::arg("iters") = 50);
#else
  m.attr("has_cuda") = false;
  m.def("cuda_device_available", [] { return false; });
#endif
}
