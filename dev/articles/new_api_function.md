# Adding an API Function

Here, we will explain how to add an API function to the {anvil} package.
API functions are the main user facing API and are follow the `nv_*`
naming scheme. There are two types of API functions:

1.  Thin wrappers around primitives that add convenience like type
    casting and scalar broadcasting (AP function `nv_add` is a wrapper
    around the primitive `nvl_add`).
2.  Common mathematical functions that are expressed in terms of
    primitives (such as `nv_rnorm`).

Here, we will focus on contributing such an API function to the {anvil}
package. Specifically, API functions in the {anvil} package should:

1.  Work with any backend.
2.  Accept the following argument types for their dynamic inputs:
    1.  `numeric` and `logical` length-1 vectors
    2.  `numeric` and `logical` R arrays
    3.  `AnvilArray`s
3.  Always output `AnvilArray`s

## 1. Work with any backend

The first requirement means that a user of the {anvil} package should be
able to use an {anvil} function with `AnvilArray`s from both the *xla*
and the *quickr* backend. When you are just calling into other anvil
primitives or correctly created API functions, there is nothing to take
care of. This requirement is only relevant, when the API function is
[`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md)ted (or
internally calls into such a function). For example, we might want to
add an API function that adds three to a tensor and jit the function. By
setting the `backend` argument to `"auto"`, we can use it with both
backends:

``` r
library(anvil)
nv_add_three_good <- jit(\(x) x + 3, backend = "auto")
nv_add_three_good(nv_scalar(1, backend = "xla"))
#> AnvilArray
#>  4
#> [ CPUf32{} ]
```

``` r
nv_add_three_good(nv_scalar(1, backend = "quickr"))
#> AnvilArray
#> [1] 4
#> [ CPUf64{} ]
```

Otherwise, the function would fail in eager mode when provided with an
input from a different backend.

``` r
nv_add_three_bad <- jit(\(x) x + 3)
```

``` r
nv_add_three_bad(nv_scalar(1, backend = "quickr"))
#> Error in `check_single_backend()`:
#> ! Cannot compile a "xla" program with inputs from other backends.
#> ℹ Found arrays from backend "quickr".
#> ℹ anvil does not support mixing backends in a single compiled program.
#> ℹ Ensure all inputs and closed-over constants use the "xla" backend.
```

Special care must also be taken when creating constants. It needs to be
ensured that the constant is created on the correct device and for the
correct backend. If we don’t specify it and do not jit the API function,
the default backend is used, which means it won’t work with non-default
backends:

``` r
nv_add_const_bad <- \(x) {
  three <- nv_scalar(3)
  three + x
}
```

``` r
nv_add_const_bad(nv_scalar(1, backend = "quickr"))
#> Error in `as_anvil_arrays()`:
#> ! Found inputs from multiple backends.
#> ℹ Found backends "xla" and "quickr".
```

If the API function was `jit`ted, than there would be no issue, because
within a [`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md)
context, the correct backend can be inferred from the context, so all we
have to do is set `backend = "auto"`.

``` r
nv_add_const_jit <- jit(nv_add_const_bad, backend = "auto")
nv_add_const_jit(nv_scalar(1, backend = "xla"))
#> AnvilArray
#>  4
#> [ CPUf32{} ]
```

If we don’t want to JIT the API function, then we need to pass the
backend of the input to the constant creation. For that, we recommend
using the `nv_<op>_like` functions that will use the attributes of the
provided example as defaults for `dtype`, `shape`, `backend`, `device`,
and `ambiguous`:

``` r
nv_add_const_eager <- \(x) {
  three <- nv_scalar(like = x, 3)
  x + three
}
```

The `nv_<op>_like` form is also the correct way to create constants that
live on the *same device* as an arrayish input. Calling
[`device()`](https://r-xla.github.io/anvil/dev/reference/device.md)
directly on the input would work for a concrete `AnvilArray`, but under
[`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md) that same
input is a `GraphBox` — and
[`device()`](https://r-xla.github.io/anvil/dev/reference/device.md) on a
`GraphBox` raises an error (tracing has no concrete device; jit handles
placement at the input/output boundary). Compare:

``` r
# BAD: errors under jit() because `device(x)` is not defined for a GraphBox
nv_tril_bad <- function(x) {
  zeros <- nv_fill(0, shape(x), dtype = dtype(x), device = device(x))
  nv_ifelse(x > 0, x, zeros)
}
jit(nv_tril_bad)(nv_array(matrix(c(-1, 2, -3, 4), 2, 2)))
#> AnvilArray
#>  0 0
#>  2 4
#> [ CPUf32{2,2} ]
```

``` r
# GOOD: `nv_fill_like` reads dtype/shape/device from `x`, and under jit()
# it picks the device up from the tracing context instead of calling `device()`.
nv_tril_good <- function(x) {
  nv_ifelse(x > 0, x, nv_fill_like(x, 0))
}
jit(nv_tril_good)(nv_array(matrix(c(-1, 2, -3, 4), 2, 2)))
#> AnvilArray
#>  0 0
#>  2 4
#> [ CPUf32{2,2} ]
```

## 2. Input conversion

API functions accept R literals (length-1 vectors, R arrays),
`AnvilArray`s, and (during tracing) graph boxes. Primitives handle all
of these, but having the same arrayish value arrive as three different R
types makes the API function body awkward: every branch would have to
re-check which case it is in. To keep the function body simple, we
standardize at the top – after that, we can treat every argument as an
`AnvilArray` (or a tracing box) without caring about the original R
type. Anvil provides two helpers for this:

- `as_anvil_array(x, device = NULL)` – use for API functions that take a
  **single** array input. It converts an R value to an `AnvilArray` on
  the given device (or the default device if `NULL`), and passes
  `AnvilArray`s through.
- `as_anvil_arrays(...)` – use for API functions with **multiple** array
  inputs. It infers a common device from any concrete `AnvilArray` in
  `...`, places R literals on that device, and errors if the concrete
  inputs live on different devices (or backends).

Both helpers short-circuit during tracing: inside
[`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md), device
placement is already handled at the input/output boundary and when
boxing constants.

### Single input

``` r
nv_expand <- function(x, shape) {
  x <- as_anvil_array(x)
  if (identical(shape(x), shape)) {
    return(x)
  }
  nv_broadcast_to(x, shape)
}
```

### Multiple inputs

Use
[`as_anvil_arrays()`](https://r-xla.github.io/anvil/dev/reference/as_anvil_array.md)
to normalize all arrayish inputs in one call:

``` r
nv_add_scaled <- function(x, y, alpha) {
  args <- as_anvil_arrays(x, y, alpha)
  args[[1L]] + args[[2L]] * args[[3L]]
}
```

## 3. Always output anvil arrays

## Inputs that require checks must be static

Any argument whose value the API function inspects (e.g. for validation,
to pick a code path, or to compute a shape or `dim` index) must be a
**static** argument under
[`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md). Under
tracing, arrayish inputs are replaced by `GraphBox` tracers and their
concrete R value is not available – assertions on the value would either
fail or see the tracer instead of the user’s data. Static arguments, in
contrast, are embedded as constants into the compiled program, so the
function body sees their real R value during tracing.

As a rule of thumb, if your API function body contains `if`, `switch`,
`stopifnot`, `checkmate::assert_*`, or any other code that branches on
or validates a value, that argument should be static. This typically
applies to arguments like `dims`, `shape`, `dim`, flags, mode strings,
or dtype specifiers – anything that controls the *structure* of the
computation rather than participating in it as data.

``` r
nv_reduce_sum <- function(x, dims = NULL) {
  if (!is.null(dims)) {
    assert_integerish(dims, lower = 1L, upper = ndims(x))
  }
  # ...
}

# `dims` is inspected, so it must be static when jitted:
jit(nv_reduce_sum, static = "dims")(x, dims = 1L)
```

Arrayish inputs (the actual data being operated on) should *not* be
static – they are meant to be traced.
