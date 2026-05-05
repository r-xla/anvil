# anvl (development version)

## Breaking Changes

* Renamed user-facing API functions to match base R names:
  `nv_sine()` -> `nv_sin()`, `nv_cosine()` -> `nv_cos()`,
  `nv_ceil()` -> `nv_ceiling()`, `nv_cholesky()` -> `nv_chol()`.
  The corresponding primitives were renamed in step:
  `prim_sine()` -> `prim_sin()`, `prim_cosine()` -> `prim_cos()`,
  `prim_cholesky()` -> `prim_chol()`.
* `nv_reduce_mean()` was renamed to `nv_mean()`.

## New Features

* Added `AnvlArray` -> R `vector` converters `as.numeric`, `as.double`,
  `as.integer`, `as.logical` and `as.vector`.
* New API functions `nv_rbind()` and `nv_cbind()` and corresponding
  `rbind()`/`cbind()` generics.
* New cumulative primitives and API functions:
  `nv_cumsum()`, `nv_cumprod()`, `nv_cummax()`, `nv_cummin()` (and
  the corresponding `prim_*` primitives).
  `prim_cumprod()` does not yet have a reverse rule.
* Added new function `await()` that blocks until the underlying computation
  has finished.
* New tree utilities `map_tree()` and `pmap_tree()` for applying functions
  leaf-wise over (possibly nested) lists.
* New primitives: `prim_sort()`, `prim_top_k()`, `prim_reduce()`,
  `prim_argmax()`, `prim_argmin()`.
* New API functions:
  * `nv_sort()` to sort along a dimension.
  * `nv_argsort()` to return the indices that would sort the array.
  * `nv_top_k()` to return the `k` largest values along a dimension.
  * `nv_median()` to compute the median along a dimension. Also dispatches
    from base R's `median()`.
  * `nv_quantile()` to compute quantiles along a dimension.
  * `nv_argmax()` and `nv_argmin()` to find the index of the maximum/minimum
    along a dimension. Ties are broken by returning the smallest index.
  * `nv_select()` to select a slice along a dimension by index.
  * `nv_flatten()` for flattening a vector
* `mean()` and `median()` now error when called with `na.rm = TRUE`, since
  anvl arrays do not carry `NA`s. `mean()` also rejects non-zero `trim`.

## Other

* `nv_reduce_sum()`, `nv_reduce_prod()`, `nv_reduce_max()`, `nv_reduce_min()`,
  `nv_reduce_any()`, `nv_reduce_all()` and `nv_mean()` now default
  `dims = NULL`, which reduces over all dimensions and returns a scalar.
  Previously, `dims` was required.

# anvl 0.2.0

## Breaking Changes

* The package was renamed from `anvil` to `anvl` to avoid a conflict
  with the Bioconductor package `AnVIL`.
* `AnvilTensor`/`nv_tensor` were renamed to `AnvlArray` and `nv_array` to be
  more in line with R's `array()`.
  Also, `nv_aten()` was renamed to `nv_aval()`.
* Subsetting with `list()` (e.g. `x[list(1, 3)]`) is no longer supported.
  Use `array()` to wrap the indices instead, e.g. `x[array(c(1L, 3L))]`.
  This mirrors the input convention used everywhere else in the package.
* Removed *debug mode*.
* Remove NSE support for `nvl_if`. It now requires passing 0-argument
  closures as `true` and `false` arguments.
* Primitives renamed from `nvl_*` to `prim_*`.
  The underlying primitive object containing the rules and metadata
  is now part of the `JitPrimitive` function via the `primitive` attribute.

## New Features

* Better composability:
  `jit()`ted functions can now be used in other `jit()`-calls.
  This is the mechanism underlying the new *eager mode*.
* *Eager mode* was added:
  This means, you can now do `nv_add(1, nv_array(1:2))` and it will
  actually perform the computation and not only do type inference.
* An experimental [{quickr}](https://github.com/t-kalinowski/quickr) backend was added
  It only runs on CPU for now and supports a subset of available operations.
  You can enable it via the `backend` argument in `jit()` and
  `nv_array()` or via the `anvl.default_backend` option.
* New primitives:
  * `nvl_cholesky()` to compute the Cholesky decomposition of a matrix.
  * `nvl_triangular_solve()` to solve a system of linear equations with a triangular matrix.
* New API functions (+ corresponding R generic implementations):
  * `nv_diag()` to create a diagonal matrix from a 1-D tensor.
  * `nv_eye()` to create an identity matrix.
  * `nv_solve()` to solve a system of linear equations.
  * `nv_cholesky()` to compute the Cholesky decomposition of a matrix.
  * `nv_device()` constructs a backend-specific device object (e.g. `nv_device("cpu")`)
    that can be passed as `device` to array constructors like `nv_fill()` or `nv_iota()`.
  * `nv_crossprod()` and `nv_tcrossprod()` for matrix cross-products.
  * `nv_outer()` for the outer product.
  * `nv_extract_diag()` to extract the diagonal of a matrix.
  * `nv_trace()` to compute the trace of a matrix.
  * `nv_tril()` and `nv_triu()` to extract lower/upper triangular parts.
  * `nv_squeeze()` and `nv_unsqueeze()` to drop or add length-1 dimensions.
  * `nv_log2()` and `nv_log10()`.
  * `nv_is_infinite()` and `nv_is_nan()`.
  * `nv_sd()` and `nv_var()` for standard deviation and variance.
* `jit()` now accepts integer positions for the `static` argument.
* New S3 methods `dim()`, `nrow()`, `ncol()`, and `length()` for anvl arrays.
* Printing tensors via `nv_print()` now also works on GPUs.
* R vectors of length 1 and arrays are now auto-converted when being passed
  to `jit`ted functions.
* Improved device handling in `jit()`

## Performance

* Many operations are now done asynchronously, which improves performance,
  especially on GPUs.

## Bug Fixes

* +-Inf/NaN are correctly created for `f64` when inlined into the XLA exectuable (#182).
  This caused wrong results with e.g. `nv_reduce_max()` when working with `f64`.
* Corrected argument checks in `nv_iota()`.
* Fix check that `wrt` arguments in `gradient()` must be floats.
* `nv_subset()` and `nv_subset_assign()` now error on trailing-comma subscripts (#273).

## Documentation

* New vignette on implementing Gaussian Processes.
* New vignette on implementing Metropolis-Hastings sampling.

## Platform support and installation

* An installation guide was added.
* Linux on ARM is now supported (CPU only).
* To use the CUDA backend, it is now possible to install the `cuda12.8`
  package (see installation guide), which only requires a compatible CUDA
  driver.

# anvl 0.1.0

Initial release
