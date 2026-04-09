#include "xla/ffi/api/ffi.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

// LAPACK declarations (Fortran calling convention)
extern "C" {
void sgeqrf_(const int* m, const int* n, float* a, const int* lda, float* tau,
             float* work, const int* lwork, int* info);
void dgeqrf_(const int* m, const int* n, double* a, const int* lda,
             double* tau, double* work, const int* lwork, int* info);
void sorgqr_(const int* m, const int* n, const int* k, float* a,
             const int* lda, const float* tau, float* work, const int* lwork,
             int* info);
void dorgqr_(const int* m, const int* n, const int* k, double* a,
             const int* lda, const double* tau, double* work, const int* lwork,
             int* info);
}

using namespace xla::ffi;

template <typename T>
static void lapack_geqrf(int m, int n, T* a, int lda, T* tau, T* work,
                          int lwork, int* info);

template <>
void lapack_geqrf<float>(int m, int n, float* a, int lda, float* tau,
                          float* work, int lwork, int* info) {
  sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
}

template <>
void lapack_geqrf<double>(int m, int n, double* a, int lda, double* tau,
                           double* work, int lwork, int* info) {
  dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
}

template <typename T>
static void lapack_orgqr(int m, int n, int k, T* a, int lda, const T* tau,
                          T* work, int lwork, int* info);

template <>
void lapack_orgqr<float>(int m, int n, int k, float* a, int lda,
                          const float* tau, float* work, int lwork,
                          int* info) {
  sorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info);
}

template <>
void lapack_orgqr<double>(int m, int n, int k, double* a, int lda,
                           const double* tau, double* work, int lwork,
                           int* info) {
  dorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info);
}

// QR decomposition: A (m x n) -> Q (m x k), R (k x n), where k = min(m, n).
//
// All buffers use column-major layout (specified via operand_layouts and
// result_layouts on the custom_call). XLA handles the row-major <-> column-major
// conversion transparently — the handler works directly with LAPACK-native layout.
template <typename T>
static Error qr_impl(AnyBuffer input, Result<AnyBuffer> q_out,
                      Result<AnyBuffer> r_out) {
  auto dims = input.dimensions();
  int m = static_cast<int>(dims[0]);
  int n = static_cast<int>(dims[1]);
  int k = std::min(m, n);

  // Copy input to working buffer (geqrf overwrites in place).
  std::vector<T> a(m * n);
  std::memcpy(a.data(), input.untyped_data(), m * n * sizeof(T));

  // Workspace query for geqrf
  std::vector<T> tau(k);
  int lwork = -1;
  T work_size;
  int info;
  lapack_geqrf<T>(m, n, a.data(), m, tau.data(), &work_size, lwork, &info);
  if (info != 0) {
    return Error::Internal("LAPACK geqrf workspace query failed");
  }

  lwork = static_cast<int>(work_size);
  std::vector<T> work(lwork);

  // Compute QR factorization
  lapack_geqrf<T>(m, n, a.data(), m, tau.data(), work.data(), lwork, &info);
  if (info != 0) {
    return Error::Internal("LAPACK geqrf failed with info = " +
                           std::to_string(info));
  }

  // Extract R: upper triangular of first k rows, column-major [k x n].
  T* r_data = static_cast<T*>((*r_out).untyped_data());
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < k; i++) {
      r_data[j * k + i] = (i <= j) ? a[j * m + i] : T(0);
    }
  }

  // Compute Q from Householder reflectors.
  // orgqr overwrites the first k columns of a with Q [m x k].
  lwork = -1;
  lapack_orgqr<T>(m, k, k, a.data(), m, tau.data(), &work_size, lwork, &info);
  if (info != 0) {
    return Error::Internal("LAPACK orgqr workspace query failed");
  }

  lwork = static_cast<int>(work_size);
  work.resize(lwork);
  lapack_orgqr<T>(m, k, k, a.data(), m, tau.data(), work.data(), lwork, &info);
  if (info != 0) {
    return Error::Internal("LAPACK orgqr failed with info = " +
                           std::to_string(info));
  }

  // Q is the first k columns of a, already column-major [m x k].
  T* q_data = static_cast<T*>((*q_out).untyped_data());
  std::memcpy(q_data, a.data(), m * k * sizeof(T));

  return Error::Success();
}

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
