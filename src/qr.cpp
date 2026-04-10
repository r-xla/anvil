#include "xla/ffi/api/ffi.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
void dgeqrf_(const int* m, const int* n, double* a, const int* lda,
             double* tau, double* work, const int* lwork, int* info);
void dorgqr_(const int* m, const int* n, const int* k, double* a,
             const int* lda, const double* tau, double* work, const int* lwork,
             int* info);
#ifndef _WIN32
void sgeqrf_(const int* m, const int* n, float* a, const int* lda, float* tau,
             float* work, const int* lwork, int* info);
void sorgqr_(const int* m, const int* n, const int* k, float* a,
             const int* lda, const float* tau, float* work, const int* lwork,
             int* info);
#endif
}

using namespace xla::ffi;

// QR decomposition: A (m x n) -> Q (m x k), R (k x n), where k = min(m, n).
//
// All buffers use column-major layout (specified via operand_layouts and
// result_layouts on the custom_call). XLA handles the row-major <-> column-major
// conversion transparently — the handler works directly with LAPACK-native layout.
template <typename T>
static Error qr_impl(AnyBuffer input, Result<AnyBuffer> q_out,
                      Result<AnyBuffer> r_out);

template <>
Error qr_impl<double>(AnyBuffer input, Result<AnyBuffer> q_out,
                       Result<AnyBuffer> r_out) {
  auto dims = input.dimensions();
  int m = static_cast<int>(dims[0]);
  int n = static_cast<int>(dims[1]);
  int k = std::min(m, n);

  std::vector<double> a(static_cast<const double*>(input.untyped_data()),
                        static_cast<const double*>(input.untyped_data()) + m * n);

  std::vector<double> tau(k);
  int lwork = -1;
  double work_size;
  int info;
  dgeqrf_(&m, &n, a.data(), &m, tau.data(), &work_size, &lwork, &info);
  if (info != 0) return Error::Internal("LAPACK geqrf workspace query failed");

  lwork = static_cast<int>(work_size);
  std::vector<double> work(lwork);
  dgeqrf_(&m, &n, a.data(), &m, tau.data(), work.data(), &lwork, &info);
  if (info != 0) {
    return Error::Internal("LAPACK geqrf failed with info = " +
                           std::to_string(info));
  }

  auto* r_data = static_cast<double*>((*r_out).untyped_data());
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < k; i++) {
      r_data[j * k + i] = (i <= j) ? a[j * m + i] : 0.0;
    }
  }

  lwork = -1;
  dorgqr_(&m, &k, &k, a.data(), &m, tau.data(), &work_size, &lwork, &info);
  if (info != 0) return Error::Internal("LAPACK orgqr workspace query failed");

  lwork = static_cast<int>(work_size);
  work.resize(lwork);
  dorgqr_(&m, &k, &k, a.data(), &m, tau.data(), work.data(), &lwork, &info);
  if (info != 0) {
    return Error::Internal("LAPACK orgqr failed with info = " +
                           std::to_string(info));
  }

  std::memcpy((*q_out).untyped_data(), a.data(), m * k * sizeof(double));
  return Error::Success();
}

#ifdef _WIN32
// Windows: R bundles its own Rlapack.dll (since there is no system LAPACK),
// which only includes double-precision routines. On macOS/Linux, R links
// against system LAPACK (Accelerate, OpenBLAS, etc.) which has both.
// So on Windows we promote f32 -> f64, call LAPACK, and demote back.
template <>
Error qr_impl<float>(AnyBuffer input, Result<AnyBuffer> q_out,
                      Result<AnyBuffer> r_out) {
  auto dims = input.dimensions();
  int m = static_cast<int>(dims[0]);
  int n = static_cast<int>(dims[1]);
  int k = std::min(m, n);

  const auto* in_f32 = static_cast<const float*>(input.untyped_data());
  std::vector<double> a(in_f32, in_f32 + m * n);

  std::vector<double> tau(k);
  int lwork = -1;
  double work_size;
  int info;
  dgeqrf_(&m, &n, a.data(), &m, tau.data(), &work_size, &lwork, &info);
  if (info != 0) return Error::Internal("LAPACK geqrf workspace query failed");

  lwork = static_cast<int>(work_size);
  std::vector<double> work(lwork);
  dgeqrf_(&m, &n, a.data(), &m, tau.data(), work.data(), &lwork, &info);
  if (info != 0) {
    return Error::Internal("LAPACK geqrf failed with info = " +
                           std::to_string(info));
  }

  std::vector<double> r_f64(k * n);
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < k; i++) {
      r_f64[j * k + i] = (i <= j) ? a[j * m + i] : 0.0;
    }
  }

  lwork = -1;
  dorgqr_(&m, &k, &k, a.data(), &m, tau.data(), &work_size, &lwork, &info);
  if (info != 0) return Error::Internal("LAPACK orgqr workspace query failed");

  lwork = static_cast<int>(work_size);
  work.resize(lwork);
  dorgqr_(&m, &k, &k, a.data(), &m, tau.data(), work.data(), &lwork, &info);
  if (info != 0) {
    return Error::Internal("LAPACK orgqr failed with info = " +
                           std::to_string(info));
  }

  auto* q_f32 = static_cast<float*>((*q_out).untyped_data());
  auto* r_f32 = static_cast<float*>((*r_out).untyped_data());
  for (int i = 0; i < m * k; i++) q_f32[i] = static_cast<float>(a[i]);
  for (int i = 0; i < k * n; i++) r_f32[i] = static_cast<float>(r_f64[i]);

  return Error::Success();
}
#else
// macOS/Linux: native single-precision LAPACK is available.
template <>
Error qr_impl<float>(AnyBuffer input, Result<AnyBuffer> q_out,
                      Result<AnyBuffer> r_out) {
  auto dims = input.dimensions();
  int m = static_cast<int>(dims[0]);
  int n = static_cast<int>(dims[1]);
  int k = std::min(m, n);

  std::vector<float> a(static_cast<const float*>(input.untyped_data()),
                       static_cast<const float*>(input.untyped_data()) + m * n);

  std::vector<float> tau(k);
  int lwork = -1;
  float work_size;
  int info;
  sgeqrf_(&m, &n, a.data(), &m, tau.data(), &work_size, &lwork, &info);
  if (info != 0) return Error::Internal("LAPACK geqrf workspace query failed");

  lwork = static_cast<int>(work_size);
  std::vector<float> work(lwork);
  sgeqrf_(&m, &n, a.data(), &m, tau.data(), work.data(), &lwork, &info);
  if (info != 0) {
    return Error::Internal("LAPACK geqrf failed with info = " +
                           std::to_string(info));
  }

  auto* r_data = static_cast<float*>((*r_out).untyped_data());
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < k; i++) {
      r_data[j * k + i] = (i <= j) ? a[j * m + i] : 0.0f;
    }
  }

  lwork = -1;
  sorgqr_(&m, &k, &k, a.data(), &m, tau.data(), &work_size, &lwork, &info);
  if (info != 0) return Error::Internal("LAPACK orgqr workspace query failed");

  lwork = static_cast<int>(work_size);
  work.resize(lwork);
  sorgqr_(&m, &k, &k, a.data(), &m, tau.data(), work.data(), &lwork, &info);
  if (info != 0) {
    return Error::Internal("LAPACK orgqr failed with info = " +
                           std::to_string(info));
  }

  std::memcpy((*q_out).untyped_data(), a.data(), m * k * sizeof(float));
  return Error::Success();
}
#endif

static Error do_qr(AnyBuffer input, Result<AnyBuffer> q_out,
                    Result<AnyBuffer> r_out) {
  switch (input.element_type()) {
    case DataType::F32:
      return qr_impl<float>(input, q_out, r_out);
    case DataType::F64:
      return qr_impl<double>(input, q_out, r_out);
    default:
      return Error::InvalidArgument(
          "QR decomposition only supports f32 and f64");
  }
}

XLA_FFI_DEFINE_HANDLER(qr_handler, do_qr,
                       Ffi::Bind()
                           .Arg<AnyBuffer>()   // input matrix (column-major)
                           .Ret<AnyBuffer>()    // Q output (column-major)
                           .Ret<AnyBuffer>());  // R output (column-major)

extern "C" {
void* anvil_get_qr_handler(void) { return (void*)qr_handler; }
}
