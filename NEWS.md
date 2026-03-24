# anvil (development version)

## Breaking Changes

* `AnvilTensor`/`nv_tensor` were renamed to `AnvilArray` and `nv_array` to be
  more in line with R's `array()`.

## New Features

* An experimental [{quickr}](https://github.com/t-kalinowski/quickr) backend is now available.
  It only runs on CPU for now and supports a subset of available operations.
  You can enable it globally via the `anvil.backend` option.
* New primitives:
  * `nvl_cholesky()` to compute the Cholesky decomposition of a matrix.
  * `nvl_triangular_solve()` to solve a system of linear equations with a triangular matrix.
* New API functions:
  * `nv_diag()` to create a diagonal matrix from a 1-D array.
  * `nv_eye()` to create an identity matrix.
  * `nv_solve()` to solve a system of linear equations.
  * `nv_cholesky()` to compute the Cholesky decomposition of a matrix.
* Improved semantics:
  * `nvl_cholesky()` now zeros out the upper/lower triangle of the output.
* Printing arrays via `nv_print()` now also works on GPUs.

## Performance

* Many operations are now done asynchronously, which improves performance,
  especially on GPUs.

## Bug Fixes

* +-Inf/NaN are correctly created for `f64` when inlined into the XLA exectuable (#182).
  This caused wrong results with e.g. `nv_reduce_max()` when working with `f64`.
* Corrected argument checks in `nv_iota()`.

## Documentation

* New vignette on implementing Gaussian Processes.
* New vignette on implementing Metropolis-Hastings sampling.
* A installation guide was added.

## Other

* To construct booleans, we now support `pjrt_buffer(..., dtype = "bool")`.
  Also `bool` is used in the printer (instead of `i1`).

# anvil 0.1.0

Initial release
