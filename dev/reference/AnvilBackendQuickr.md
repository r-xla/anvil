# Quickr backend

Constructs the quickr backend, which stores array data as plain R arrays
and compiles jitted functions to R code via the
[quickr](https://CRAN.R-project.org/package=quickr) package.

## Usage

``` r
AnvilBackendQuickr()
```

## Value

An
[`AnvilBackend`](https://r-xla.github.io/anvil/dev/reference/AnvilBackend.md)
object with subclass `"AnvilBackendQuickr"`.

## Details

To use it, the `"quickr"` package needs to be installed.

Registered automatically under the name `"quickr"` when the package is
loaded; call
[`local_backend("quickr")`](https://r-xla.github.io/anvil/dev/reference/local_backend.md)
or
[`with_backend("quickr", ...)`](https://r-xla.github.io/anvil/dev/reference/with_backend.md)
to use it. Requires the quickr package to be installed.

## Data representation

An
[`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)
with `backend = "quickr"` is, under the hood, a plain R vector or array
(`numeric`, `integer`, or `logical`) stored in the `$data` field.
[`as_array()`](https://r-xla.github.io/anvil/dev/reference/as_array.md)
returns the underlying vector/array directly without copying, and
[`nv_array()`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)
simply wraps an R vector/array. As a consequence, there is no separate
notion of a device: data always lives in R's memory and computation
always runs on the CPU.

## Status

This backend is **experimental** and has a number of limitations:

- Compilation (tracing + quickr lowering) is somewhat slow, so it is
  best suited to long-running or repeatedly-called functions where the
  one-time compilation cost is amortized.

- Only a subset of the primitives that the XLA backend supports are
  currently lowered to quickr code.

- Only the data types `f64`, `i32`, and `bool` are supported.

- Only CPU execution is supported.

## Quickr JIT arguments

- `unwrap` (`logical(1)`, default `FALSE`): if `TRUE`, the compiled
  function returns plain R arrays instead of
  [`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)s.
  Useful when the jitted function's output is consumed by non-anvil R
  code and the extra wrapping would only get stripped again.

## See also

[`AnvilBackend()`](https://r-xla.github.io/anvil/dev/reference/AnvilBackend.md),
[`AnvilBackendXla()`](https://r-xla.github.io/anvil/dev/reference/AnvilBackendXla.md),
[`local_backend()`](https://r-xla.github.io/anvil/dev/reference/local_backend.md),
[`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md).
