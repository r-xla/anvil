# JIT compile a function

Wraps a function so that it is traced and compiled on first call.
Subsequent calls with the same input structure, shapes, and dtypes hit
an LRU cache and skip recompilation. Unlike
[`xla()`](https://r-xla.github.io/anvil/dev/reference/xla.md), the
compiled executable is not created eagerly but lazily on the first
invocation.

## Usage

``` r
jit(
  f,
  static = character(),
  cache_size = 100L,
  backend = default_backend(),
  ...
)
```

## Arguments

- f:

  (`function`)  
  Function to compile. Must accept and return
  [`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)s
  (and/or static arguments).

- static:

  ([`character()`](https://rdrr.io/r/base/character.html) \|
  [`integer()`](https://rdrr.io/r/base/integer.html))  
  Names or positions of parameters of `f` that are *not* arrays. Static
  values are embedded as constants in the compiled program; a new
  compilation is triggered whenever a static value changes. For example
  useful when you want R control flow in your function.

- cache_size:

  (`integer(1)`)  
  Maximum number of compiled executables to keep in the LRU cache.

- backend:

  (`character(1)`)  
  Compilation backend. `"xla"` (default) uses PJRT/XLA. `"quickr"` uses
  [`quickr::quick()`](https://rdrr.io/pkg/quickr/man/quick.html). If
  omitted, the default comes from
  [`default_backend()`](https://r-xla.github.io/anvil/dev/reference/default_backend.md).

- ...:

  Backend-specific options. Passing an option that is not supported by
  the selected backend raises an error. See the **XLA JIT arguments**
  and **Quickr JIT arguments** sections below for the options accepted
  by each backend.

## Value

A `JitFunction` with the same formals as `f`. The returned wrapper
expects
[`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)
inputs and returns
[`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)
values (unless `unwrap = TRUE` is passed to the `"quickr"` backend).

(`function`)

## XLA JIT arguments

- `donate` ([`character()`](https://rdrr.io/r/base/character.html),
  default [`character()`](https://rdrr.io/r/base/character.html)): names
  of arguments whose underlying buffers may be donated to (i.e.,
  reused/consumed by) the compiled XLA executable. Donated buffers must
  not be used again by the caller after the call; this can reduce memory
  usage and copies for large inputs. Must not overlap with `static`.

- `device` (`NULL` \| `character(1)` \|
  [`pjrt::PJRTDevice`](https://r-xla.github.io/pjrt/reference/as_pjrt_device.html),
  default `NULL`): target device (e.g. `"cpu"`, `"cuda"`) on which the
  function is compiled and executed. When `NULL`, the device is inferred
  from the inputs; if inputs live on different devices an error is
  raised.

## Quickr JIT arguments

- `unwrap` (`logical(1)`, default `FALSE`): if `TRUE`, the compiled
  function returns plain R arrays instead of
  [`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)s.
  Useful when the jitted function's output is consumed by non-anvil R
  code and the extra wrapping would only get stripped again.

## See also

[`xla()`](https://r-xla.github.io/anvil/dev/reference/xla.md) for
ahead-of-time compilation,
[`jit_eval()`](https://r-xla.github.io/anvil/dev/reference/jit_eval.md)
for evaluating an expression once.

## Examples

``` r
f <- jit(function(x, y) x + y)
f(nv_array(1), nv_array(2))
#> AnvilArray
#>  3
#> [ CPUf32{1} ] 

# Static arguments enable data-dependent control flow
g <- jit(function(x, flag) {
  if (flag) x + 1 else x * 2
}, static = "flag")
g(nv_array(3), TRUE)
#> AnvilArray
#>  4
#> [ CPUf32{1} ] 
g(nv_array(3), FALSE)
#> AnvilArray
#>  6
#> [ CPUf32{1} ] 
with_backend("quickr", {
  h <- jit(function(x, y) x + y)
  h(nv_array(1), nv_array(2))
})
#> AnvilArray
#> [1] 3
#> [ CPUf64{1} ] 
```
