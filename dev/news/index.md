# Changelog

## anvl (development version)

### Breaking Changes

- Renamed user-facing API functions to match base R names: `nv_sine()`
  -\>
  [`nv_sin()`](https://r-xla.github.io/anvl/dev/reference/nv_sin.md),
  `nv_cosine()` -\>
  [`nv_cos()`](https://r-xla.github.io/anvl/dev/reference/nv_cos.md),
  `nv_ceil()` -\>
  [`nv_ceiling()`](https://r-xla.github.io/anvl/dev/reference/nv_ceiling.md),
  `nv_cholesky()` -\>
  [`nv_chol()`](https://r-xla.github.io/anvl/dev/reference/nv_chol.md).
  The underlying `prim_*` primitives keep their StableHLO-aligned names.

### New Features

- Added `AnvlArray` -\> R `vector` converters `as.numeric`, `as.double`,
  `as.integer` and `as.logical`
- New API functions
  [`nv_rbind()`](https://r-xla.github.io/anvl/dev/reference/nv_bind.md)
  and
  [`nv_cbind()`](https://r-xla.github.io/anvl/dev/reference/nv_bind.md)
  and corresponding
  [`rbind()`](https://rdrr.io/r/base/cbind.html)/[`cbind()`](https://rdrr.io/r/base/cbind.html)
  generics.

## anvl 0.2.0

### Breaking Changes

- The package was renamed from `anvil` to `anvl` to avoid a conflict
  with the Bioconductor package `AnVIL`.
- `AnvilTensor`/`nv_tensor` were renamed to `AnvlArray` and `nv_array`
  to be more in line with R’s
  [`array()`](https://rdrr.io/r/base/array.html). Also, `nv_aten()` was
  renamed to
  [`nv_aval()`](https://r-xla.github.io/anvl/dev/reference/AbstractArray.md).
- Subsetting with [`list()`](https://rdrr.io/r/base/list.html)
  (e.g. `x[list(1, 3)]`) is no longer supported. Use
  [`array()`](https://rdrr.io/r/base/array.html) to wrap the indices
  instead, e.g. `x[array(c(1L, 3L))]`. This mirrors the input convention
  used everywhere else in the package.
- Removed *debug mode*.
- Remove NSE support for `nvl_if`. It now requires passing 0-argument
  closures as `true` and `false` arguments.
- Primitives renamed from `nvl_*` to `prim_*`. The underlying primitive
  object containing the rules and metadata is now part of the
  `JitPrimitive` function via the `primitive` attribute.

### New Features

- Better composability:
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)ted
  functions can now be used in other
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)-calls.
  This is the mechanism underlying the new *eager mode*.
- *Eager mode* was added: This means, you can now do
  `nv_add(1, nv_array(1:2))` and it will actually perform the
  computation and not only do type inference.
- An experimental [{quickr}](https://github.com/t-kalinowski/quickr)
  backend was added It only runs on CPU for now and supports a subset of
  available operations. You can enable it via the `backend` argument in
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) and
  [`nv_array()`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
  or via the `anvl.default_backend` option.
- New primitives:
  - `nvl_cholesky()` to compute the Cholesky decomposition of a matrix.
  - `nvl_triangular_solve()` to solve a system of linear equations with
    a triangular matrix.
- New API functions (+ corresponding R generic implementations):
  - [`nv_diag()`](https://r-xla.github.io/anvl/dev/reference/nv_diag.md)
    to create a diagonal matrix from a 1-D tensor.
  - [`nv_eye()`](https://r-xla.github.io/anvl/dev/reference/nv_eye.md)
    to create an identity matrix.
  - [`nv_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_solve.md)
    to solve a system of linear equations.
  - `nv_cholesky()` to compute the Cholesky decomposition of a matrix.
  - [`nv_device()`](https://r-xla.github.io/anvl/dev/reference/nv_device.md)
    constructs a backend-specific device object
    (e.g. `nv_device("cpu")`) that can be passed as `device` to array
    constructors like
    [`nv_fill()`](https://r-xla.github.io/anvl/dev/reference/nv_fill.md)
    or
    [`nv_iota()`](https://r-xla.github.io/anvl/dev/reference/nv_iota.md).
  - [`nv_crossprod()`](https://r-xla.github.io/anvl/dev/reference/nv_crossprod.md)
    and
    [`nv_tcrossprod()`](https://r-xla.github.io/anvl/dev/reference/nv_tcrossprod.md)
    for matrix cross-products.
  - [`nv_outer()`](https://r-xla.github.io/anvl/dev/reference/nv_outer.md)
    for the outer product.
  - [`nv_extract_diag()`](https://r-xla.github.io/anvl/dev/reference/nv_extract_diag.md)
    to extract the diagonal of a matrix.
  - [`nv_trace()`](https://r-xla.github.io/anvl/dev/reference/nv_trace.md)
    to compute the trace of a matrix.
  - [`nv_tril()`](https://r-xla.github.io/anvl/dev/reference/nv_tril.md)
    and
    [`nv_triu()`](https://r-xla.github.io/anvl/dev/reference/nv_triu.md)
    to extract lower/upper triangular parts.
  - [`nv_squeeze()`](https://r-xla.github.io/anvl/dev/reference/nv_squeeze.md)
    and
    [`nv_unsqueeze()`](https://r-xla.github.io/anvl/dev/reference/nv_unsqueeze.md)
    to drop or add length-1 dimensions.
  - [`nv_log2()`](https://r-xla.github.io/anvl/dev/reference/nv_log2.md)
    and
    [`nv_log10()`](https://r-xla.github.io/anvl/dev/reference/nv_log10.md).
  - [`nv_is_infinite()`](https://r-xla.github.io/anvl/dev/reference/nv_is_infinite.md)
    and
    [`nv_is_nan()`](https://r-xla.github.io/anvl/dev/reference/nv_is_nan.md).
  - [`nv_sd()`](https://r-xla.github.io/anvl/dev/reference/nv_sd.md) and
    [`nv_var()`](https://r-xla.github.io/anvl/dev/reference/nv_var.md)
    for standard deviation and variance.
- [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md) now
  accepts integer positions for the `static` argument.
- New S3 methods [`dim()`](https://rdrr.io/r/base/dim.html),
  [`nrow()`](https://rdrr.io/r/base/nrow.html),
  [`ncol()`](https://rdrr.io/r/base/nrow.html), and
  [`length()`](https://rdrr.io/r/base/length.html) for anvl arrays.
- Printing tensors via
  [`nv_print()`](https://r-xla.github.io/anvl/dev/reference/nv_print.md)
  now also works on GPUs.
- R vectors of length 1 and arrays are now auto-converted when being
  passed to `jit`ted functions.
- Improved device handling in
  [`jit()`](https://r-xla.github.io/anvl/dev/reference/jit.md)

### Performance

- Many operations are now done asynchronously, which improves
  performance, especially on GPUs.

### Bug Fixes

- +-Inf/NaN are correctly created for `f64` when inlined into the XLA
  exectuable ([\#182](https://github.com/r-xla/anvl/issues/182)). This
  caused wrong results with
  e.g. [`nv_reduce_max()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_max.md)
  when working with `f64`.
- Corrected argument checks in
  [`nv_iota()`](https://r-xla.github.io/anvl/dev/reference/nv_iota.md).
- Fix check that `wrt` arguments in
  [`gradient()`](https://r-xla.github.io/anvl/dev/reference/gradient.md)
  must be floats.
- [`nv_subset()`](https://r-xla.github.io/anvl/dev/reference/nv_subset.md)
  and
  [`nv_subset_assign()`](https://r-xla.github.io/anvl/dev/reference/nv_subset_assign.md)
  now error on trailing-comma subscripts
  ([\#273](https://github.com/r-xla/anvl/issues/273)).

### Documentation

- New vignette on implementing Gaussian Processes.
- New vignette on implementing Metropolis-Hastings sampling.

### Platform support and installation

- An installation guide was added.
- Linux on ARM is now supported (CPU only).
- To use the CUDA backend, it is now possible to install the `cuda12.8`
  package (see installation guide), which only requires a compatible
  CUDA driver.

## anvl 0.1.0

Initial release
