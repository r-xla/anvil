#include "xla/ffi/api/ffi.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

// CUDA QR decomposition via cuSOLVER. Only compiled on non-Windows platforms
// (CUDA is not supported on Windows R builds, and dlopen is POSIX-only).
#ifndef _WIN32

#include <dlfcn.h>

using namespace xla::ffi;

// Opaque CUDA/cuSOLVER types (avoids needing CUDA SDK headers).
using CUdeviceptr = unsigned long long;

// ---------------------------------------------------------------------------
// Dynamic loading of cuSOLVER and CUDA driver API
// ---------------------------------------------------------------------------

struct GpuLibs {
  // cuSOLVER handle management
  int (*dn_create)(void **);
  int (*dn_destroy)(void *);
  int (*dn_set_stream)(void *, void *);

  // cuSOLVER geqrf (float)
  int (*s_geqrf_bs)(void *, int, int, float *, int, int *);
  int (*s_geqrf)(void *, int, int, float *, int, float *, float *, int, int *);
  // cuSOLVER geqrf (double)
  int (*d_geqrf_bs)(void *, int, int, double *, int, int *);
  int (*d_geqrf)(void *, int, int, double *, int, double *, double *, int,
                 int *);

  // cuSOLVER orgqr (float)
  int (*s_orgqr_bs)(void *, int, int, int, const float *, int, const float *,
                    int *);
  int (*s_orgqr)(void *, int, int, int, float *, int, const float *, float *,
                 int, int *);
  // cuSOLVER orgqr (double)
  int (*d_orgqr_bs)(void *, int, int, int, const double *, int, const double *,
                    int *);
  int (*d_orgqr)(void *, int, int, int, double *, int, const double *, double *,
                 int, int *);

  // CUDA driver API
  int (*mem_alloc)(CUdeviceptr *, size_t);
  int (*mem_free)(CUdeviceptr);
  int (*memcpy_dtod)(CUdeviceptr, CUdeviceptr, size_t, void *);
  int (*memset_d8)(CUdeviceptr, unsigned char, size_t, void *);
  int (*stream_sync)(void *);

  bool loaded = false;
};

template <typename T> static T load_sym(void *lib, const char *name) {
  return reinterpret_cast<T>(dlsym(lib, name));
}

static GpuLibs &get_gpu_libs() {
  static GpuLibs g;
  if (g.loaded)
    return g;

  void *cusolver = dlopen("libcusolver.so", RTLD_LAZY);
  if (!cusolver)
    cusolver = dlopen("libcusolver.so.11", RTLD_LAZY);
  if (!cusolver)
    return g;

  void *cuda = dlopen("libcuda.so.1", RTLD_LAZY);
  if (!cuda)
    return g;

  // cuSOLVER
  g.dn_create = load_sym<decltype(g.dn_create)>(cusolver, "cusolverDnCreate");
  g.dn_destroy =
      load_sym<decltype(g.dn_destroy)>(cusolver, "cusolverDnDestroy");
  g.dn_set_stream =
      load_sym<decltype(g.dn_set_stream)>(cusolver, "cusolverDnSetStream");

  g.s_geqrf_bs =
      load_sym<decltype(g.s_geqrf_bs)>(cusolver, "cusolverDnSgeqrf_bufferSize");
  g.s_geqrf = load_sym<decltype(g.s_geqrf)>(cusolver, "cusolverDnSgeqrf");
  g.d_geqrf_bs =
      load_sym<decltype(g.d_geqrf_bs)>(cusolver, "cusolverDnDgeqrf_bufferSize");
  g.d_geqrf = load_sym<decltype(g.d_geqrf)>(cusolver, "cusolverDnDgeqrf");

  g.s_orgqr_bs =
      load_sym<decltype(g.s_orgqr_bs)>(cusolver, "cusolverDnSorgqr_bufferSize");
  g.s_orgqr = load_sym<decltype(g.s_orgqr)>(cusolver, "cusolverDnSorgqr");
  g.d_orgqr_bs =
      load_sym<decltype(g.d_orgqr_bs)>(cusolver, "cusolverDnDorgqr_bufferSize");
  g.d_orgqr = load_sym<decltype(g.d_orgqr)>(cusolver, "cusolverDnDorgqr");

  // CUDA driver
  g.mem_alloc = load_sym<decltype(g.mem_alloc)>(cuda, "cuMemAlloc_v2");
  g.mem_free = load_sym<decltype(g.mem_free)>(cuda, "cuMemFree_v2");
  g.memcpy_dtod =
      load_sym<decltype(g.memcpy_dtod)>(cuda, "cuMemcpyDtoDAsync_v2");
  g.memset_d8 = load_sym<decltype(g.memset_d8)>(cuda, "cuMemsetD8Async");
  g.stream_sync =
      load_sym<decltype(g.stream_sync)>(cuda, "cuStreamSynchronize");

  g.loaded = true;
  return g;
}

// ---------------------------------------------------------------------------
// RAII wrapper for device memory
// ---------------------------------------------------------------------------

struct DeviceMem {
  CUdeviceptr ptr = 0;
  GpuLibs &g;
  explicit DeviceMem(GpuLibs &g) : g(g) {}
  ~DeviceMem() {
    if (ptr)
      g.mem_free(ptr);
  }
  DeviceMem(const DeviceMem &) = delete;
  DeviceMem &operator=(const DeviceMem &) = delete;

  int alloc(size_t bytes) { return g.mem_alloc(&ptr, bytes); }
};

// ---------------------------------------------------------------------------
// Cached cuSOLVER handle (created once, reused across calls)
// ---------------------------------------------------------------------------

static Error ensure_handle(GpuLibs &g, void *&handle) {
  static void *cached = nullptr;
  if (!cached) {
    if (g.dn_create(&cached) != 0) {
      return Error::Internal("Failed to create cuSOLVER handle");
    }
  }
  handle = cached;
  return Error::Success();
}

// ---------------------------------------------------------------------------
// cuSOLVER trait dispatch (float vs double)
// ---------------------------------------------------------------------------

template <typename T> struct Solver;

template <> struct Solver<float> {
  static auto geqrf_bs(GpuLibs &g) { return g.s_geqrf_bs; }
  static auto geqrf(GpuLibs &g) { return g.s_geqrf; }
  static auto orgqr_bs(GpuLibs &g) { return g.s_orgqr_bs; }
  static auto orgqr(GpuLibs &g) { return g.s_orgqr; }
};

template <> struct Solver<double> {
  static auto geqrf_bs(GpuLibs &g) { return g.d_geqrf_bs; }
  static auto geqrf(GpuLibs &g) { return g.d_geqrf; }
  static auto orgqr_bs(GpuLibs &g) { return g.d_orgqr_bs; }
  static auto orgqr(GpuLibs &g) { return g.d_orgqr; }
};

// ---------------------------------------------------------------------------
// QR decomposition on GPU via cuSOLVER
// ---------------------------------------------------------------------------

template <typename T>
static Error qr_cuda_impl(void *stream, AnyBuffer input,
                          Result<AnyBuffer> q_out, Result<AnyBuffer> r_out) {
  auto &g = get_gpu_libs();
  if (!g.loaded) {
    return Error::Internal("CUDA/cuSOLVER libraries not available");
  }

  void *handle;
  auto err = ensure_handle(g, handle);
  if (!err.success())
    return err;
  g.dn_set_stream(handle, stream);

  auto dims = input.dimensions();
  int m = static_cast<int>(dims[0]);
  int n = static_cast<int>(dims[1]);
  int k = std::min(m, n);

  auto input_ptr = reinterpret_cast<CUdeviceptr>(input.untyped_data());
  auto q_ptr = reinterpret_cast<CUdeviceptr>((*q_out).untyped_data());
  auto r_ptr = reinterpret_cast<CUdeviceptr>((*r_out).untyped_data());

  // Device allocations: working copy of A, tau vector, devInfo, workspace.
  DeviceMem d_a(g), d_tau(g), d_info(g), d_work(g);
  d_a.alloc(m * n * sizeof(T));
  d_tau.alloc(k * sizeof(T));
  d_info.alloc(sizeof(int));

  // Copy input to working buffer (geqrf overwrites in place).
  g.memcpy_dtod(d_a.ptr, input_ptr, m * n * sizeof(T), stream);

  // Workspace query and allocation for geqrf.
  int lwork = 0;
  Solver<T>::geqrf_bs(g)(handle, m, n, reinterpret_cast<T *>(d_a.ptr), m,
                         &lwork);
  d_work.alloc(lwork * sizeof(T));

  // QR factorization: A -> reflectors (lower) + R (upper).
  int status = Solver<T>::geqrf(g)(handle, m, n, reinterpret_cast<T *>(d_a.ptr),
                                   m, reinterpret_cast<T *>(d_tau.ptr),
                                   reinterpret_cast<T *>(d_work.ptr), lwork,
                                   reinterpret_cast<int *>(d_info.ptr));
  if (status != 0) {
    return Error::Internal("cuSOLVER geqrf failed with status = " +
                           std::to_string(status));
  }

  // Extract R: zero the output, then copy upper triangular column by column.
  g.memset_d8(r_ptr, 0, k * n * sizeof(T), stream);
  for (int j = 0; j < n; j++) {
    int elems = std::min(j + 1, k);
    g.memcpy_dtod(r_ptr + j * k * sizeof(T), d_a.ptr + j * m * sizeof(T),
                  elems * sizeof(T), stream);
  }

  // Copy first k columns of factored A to Q output (column-major, so first
  // m*k elements), then run orgqr in-place on Q to produce the Q matrix.
  g.memcpy_dtod(q_ptr, d_a.ptr, m * k * sizeof(T), stream);

  // Workspace query for orgqr (may differ from geqrf).
  int lwork_orgqr = 0;
  Solver<T>::orgqr_bs(g)(handle, m, k, k, reinterpret_cast<const T *>(q_ptr), m,
                         reinterpret_cast<const T *>(d_tau.ptr), &lwork_orgqr);

  DeviceMem d_work2(g);
  T *work_ptr;
  if (lwork_orgqr <= lwork) {
    work_ptr = reinterpret_cast<T *>(d_work.ptr);
  } else {
    d_work2.alloc(lwork_orgqr * sizeof(T));
    work_ptr = reinterpret_cast<T *>(d_work2.ptr);
  }

  // Compute Q from Householder reflectors.
  status =
      Solver<T>::orgqr(g)(handle, m, k, k, reinterpret_cast<T *>(q_ptr), m,
                          reinterpret_cast<const T *>(d_tau.ptr), work_ptr,
                          lwork_orgqr, reinterpret_cast<int *>(d_info.ptr));
  if (status != 0) {
    return Error::Internal("cuSOLVER orgqr failed with status = " +
                           std::to_string(status));
  }

  return Error::Success();
}

static Error do_qr_cuda(void *stream, AnyBuffer input, Result<AnyBuffer> q_out,
                        Result<AnyBuffer> r_out) {
  switch (input.element_type()) {
  case DataType::F32:
    return qr_cuda_impl<float>(stream, input, q_out, r_out);
  case DataType::F64:
    return qr_cuda_impl<double>(stream, input, q_out, r_out);
  default:
    return Error::InvalidArgument("QR decomposition only supports f32 and f64");
  }
}

XLA_FFI_DEFINE_HANDLER(qr_handler_cuda, do_qr_cuda,
                       Ffi::Bind()
                           .Ctx<PlatformStream<void *>>()
                           .Arg<AnyBuffer>()
                           .Ret<AnyBuffer>()
                           .Ret<AnyBuffer>());

extern "C" {
void *anvil_get_qr_handler_cuda(void) {
  return reinterpret_cast<void *>(qr_handler_cuda);
}
}

#else // _WIN32

extern "C" {
void *anvil_get_qr_handler_cuda(void) { return nullptr; }
}

#endif
