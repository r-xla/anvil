# Changelog

## anvil (development version)

### New Features

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
- Improved semantics:
  - [`nvl_cholesky()`](https://r-xla.github.io/anvil/dev/reference/nvl_cholesky.md)
    now zeros out the upper/lower triangle of the output.

### Bug Fixes

- +-Inf/NaN are correctly created for `f64` when inlined into the XLA
  exectuable ([\#182](https://github.com/r-xla/anvil/issues/182)). This
  caused wrong results with
  e.g. [`nv_reduce_max()`](https://r-xla.github.io/anvil/dev/reference/nv_reduce_max.md)
  when working with `f64`.
- Corrected argument checks in
  [`nv_iota()`](https://r-xla.github.io/anvil/dev/reference/nv_iota.md).

### Documentation

- New vignette on implementing Gaussian Processes.
- New vignette on implementing Metropolis-Hastings sampling.

## anvil 0.1.0

Initial release
