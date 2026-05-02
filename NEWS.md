# anvil (development version)

## Breaking Changes

* `AnvilTensor`/`nv_tensor` were renamed to `AnvilArray` and `nv_array` to be
  more in line with R's `array()`.
  Also, `nv_aten()` was renamed to `nv_abstract()`.
* `nv_solve()` now uses LU decomposition with partial pivoting instead of
  Cholesky, and consequently no longer requires `a` to be symmetric
  positive-definite -- any non-singular square matrix works. The result is
  the same on SPD inputs.

## New Features

* Better composability:
  `jit()`ted functions can now be traced when used within `jit()`.
* An experimental [{quickr}](https://github.com/t-kalinowski/quickr) backend is now available.
  It only runs on CPU for now and supports a subset of available operations.
  You can enable it globally via the `backend` argument in `jit()` and
  `nv_array()` or via the `anvil.default_backend` option.
* New primitives:
  * `nvl_cholesky()` to compute the Cholesky decomposition of a matrix.
  * `nvl_triangular_solve()` to solve a system of linear equations with a triangular matrix.
  * `nvl_qr()` for QR decomposition.
  * `nvl_lu()` for LU decomposition with partial pivoting.
  * `nvl_svd()` for the reduced ("economy") singular value decomposition.
  * `nvl_eigh()` for the eigendecomposition of a symmetric matrix.
* New API functions:
  * `nv_diag()` to create a diagonal matrix from a 1-D tensor.
  * `nv_eye()` to create an identity matrix.
  * `nv_solve()` to solve a system of linear equations.
  * `nv_cholesky()` to compute the Cholesky decomposition of a matrix.
  * `nv_qr()`, `nv_lu()`, `nv_svd()`, `nv_eigh()` wrappers for the new
    primitives, returning named lists.
  * `nv_det()` and `nv_logdet()` for determinants (computed from the LU
    factors -- the latter is numerically stable for nearly-singular
    matrices).
  * `nv_inv()` for explicit matrix inverse (composes `nv_solve` with the
    identity right-hand side; prefer `nv_solve` directly for systems).
* Autodiff support for the new linear-algebra primitives:
  * `nvl_qr()` (square, tall, wide), `nvl_eigh()`, `nvl_svd()`
    (square, tall, wide), and `nvl_lu()` (square only). The reverse rules
    follow Townsend (2016) and Giles (2008); see the per-primitive
    `@references` for citations.
* S3 method dispatch for base R's linalg generics on `AnvilArray`:
  `solve()` (with `solve(a)` returning the inverse), `qr()`, `chol()`,
  and `determinant()`. Code that uses base-R syntax for these now Just
  Works inside `jit()`.
* New `nv_eigen()` -- a base-R-`eigen()`-shaped wrapper around
  `nv_eigh()` (returns `list(values, vectors)` with eigenvalues in
  descending order, matching base R). Symmetric only; passing
  `symmetric = FALSE` raises a clear error, since general
  (non-symmetric) eigendecomposition would require complex-dtype outputs
  which anvl does not yet support. `svd()` and `eigen()` are not S3
  generics in base R, so we don't shadow them; use `nv_svd()` /
  `nv_eigen()` directly.
* Improved semantics:
  * `nvl_cholesky()` now zeros out the upper/lower triangle of the output.
* Printing tensors via `nv_print()` now also works on GPUs.

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
* An installation guide was added.

## Miscellaneous

* Linux on ARM is now supported (CPU only).
* To use the CUDA backend, it is now possible to install the `cuda12.8`
  package (see installation guide), which only requires a compatible CUDA
  driver.



# anvil 0.1.0

Initial release
