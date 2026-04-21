# Changelog

## anvil (development version)

### Breaking Changes

- `AnvilTensor`/`nv_tensor` were renamed to `AnvilArray` and `nv_array`
  to be more in line with R’s
  [`array()`](https://rdrr.io/r/base/array.html). Also, `nv_aten()` was
  renamed to
  [`nv_aval()`](https://r-xla.github.io/anvil/dev/reference/AbstractArray.md).
- Subsetting with [`list()`](https://rdrr.io/r/base/list.html)
  (e.g. `x[list(1, 3)]`) is no longer supported. Use
  [`array()`](https://rdrr.io/r/base/array.html) to wrap the indices
  instead, e.g. `x[array(c(1L, 3L))]`. This mirrors the input convention
  used everywhere else in the package.
- Removed *debug mode*.
- Remove NSE support for `nvl_if`. It now requires passing 0-argument
  closures as `true` and `false` arguments.

### New Features

- Better composability:
  [`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md)ted
  functions can now be used in other
  [`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md)-calls.
  This is the mechanism underlying the new *eager mode*.
- *Eager mode* was added: This means, you can now do
  `nv_add(1, nv_array(1:2))` and it will actually perform the
  computation and not only do type inference.
- An experimental [{quickr}](https://github.com/t-kalinowski/quickr)
  backend was added It only runs on CPU for now and supports a subset of
  available operations. You can enable it via the `backend` argument in
  [`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md) and
  [`nv_array()`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)
  or via the `anvil.default_backend` option.
- New primitives:
  - [`nvl_cholesky()`](https://r-xla.github.io/anvil/dev/reference/nvl_cholesky.md)
    to compute the Cholesky decomposition of a matrix.
  - [`nvl_triangular_solve()`](https://r-xla.github.io/anvil/dev/reference/nvl_triangular_solve.md)
    to solve a system of linear equations with a triangular matrix.
- New API functions:
  - [`nv_diag()`](https://r-xla.github.io/anvil/dev/reference/nv_diag.md)
    to create a diagonal matrix from a 1-D tensor.
  - [`nv_eye()`](https://r-xla.github.io/anvil/dev/reference/nv_eye.md)
    to create an identity matrix.
  - [`nv_solve()`](https://r-xla.github.io/anvil/dev/reference/nv_solve.md)
    to solve a system of linear equations.
  - [`nv_cholesky()`](https://r-xla.github.io/anvil/dev/reference/nv_cholesky.md)
    to compute the Cholesky decomposition of a matrix.
  - [`nv_device()`](https://r-xla.github.io/anvil/dev/reference/nv_device.md)
    constructs a backend-specific device object
    (e.g. `nv_device("cpu")`) that can be passed as `device` to array
    constructors like
    [`nv_fill()`](https://r-xla.github.io/anvil/dev/reference/nv_fill.md)
    or
    [`nv_iota()`](https://r-xla.github.io/anvil/dev/reference/nv_iota.md).
- New S3 methods [`dim()`](https://rdrr.io/r/base/dim.html),
  [`nrow()`](https://rdrr.io/r/base/nrow.html),
  [`ncol()`](https://rdrr.io/r/base/nrow.html), and
  [`length()`](https://rdrr.io/r/base/length.html) for anvil arrays.
- Printing tensors via
  [`nv_print()`](https://r-xla.github.io/anvil/dev/reference/nv_print.md)
  now also works on GPUs.
- R vectors of length 1 and arrays are now auto-converted when being
  passed to `jit`ted functions.
- Improved device handling in
  [`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md)

### Performance

- Many operations are now done asynchronously, which improves
  performance, especially on GPUs.

### Bug Fixes

- +-Inf/NaN are correctly created for `f64` when inlined into the XLA
  exectuable ([\#182](https://github.com/r-xla/anvil/issues/182)). This
  caused wrong results with
  e.g. [`nv_reduce_max()`](https://r-xla.github.io/anvil/dev/reference/nv_reduce_max.md)
  when working with `f64`.
- Corrected argument checks in
  [`nv_iota()`](https://r-xla.github.io/anvil/dev/reference/nv_iota.md).
- Fix check that `wrt` arguments in
  [`gradient()`](https://r-xla.github.io/anvil/dev/reference/gradient.md)
  must be floats.

### Documentation

- New vignette on implementing Gaussian Processes.
- New vignette on implementing Metropolis-Hastings sampling.

### Platform support and installation

- An installation guide was added.
- Linux on ARM is now supported (CPU only).
- To use the CUDA backend, it is now possible to install the `cuda12.8`
  package (see installation guide), which only requires a compatible
  CUDA driver.

## anvil 0.1.0

Initial release
