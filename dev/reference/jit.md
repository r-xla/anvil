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
  donate = character(),
  device = NULL,
  backend = default_backend()
)
```

## Arguments

- f:

  (`function`)  
  Function to compile. Must accept and return
  [`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)s
  (and/or static arguments).

- static:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of parameters of `f` that are *not* arrays. Static values are
  embedded as constants in the compiled program; a new compilation is
  triggered whenever a static value changes. For example useful when you
  want R control flow in your function.

- cache_size:

  (`integer(1)`)  
  Maximum number of compiled executables to keep in the LRU cache.

- donate:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of the arguments whose buffers should be donated. Donated
  buffers can be aliased with outputs of the same type, allowing
  in-place operations and reducing memory usage. An argument cannot
  appear in both `donate` and `static`.

- device:

  (`NULL` \| `character(1)` \|
  [`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html))  
  The device to use if it cannot be inferred from the inputs or
  constants. Defaults to `"cpu"`. Only supported for `backend = "xla"`.

- backend:

  (`character(1)`)  
  Compilation backend. `"xla"` (default) uses PJRT/XLA. `"quickr"` uses
  [`quickr::quick()`](https://rdrr.io/pkg/quickr/man/quick.html). If
  omitted, the default comes from
  [`default_backend()`](https://r-xla.github.io/anvil/dev/reference/default_backend.md).

## Value

A `JitFunction` with the same formals as `f`. The returned wrapper
expects
[`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)
inputs and returns
[`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)
values.

(`function`)

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
local_backend("quickr")
h <- jit(function(x, y) x + y)
h(nv_array(1), nv_array(2))
#> AnvilArray
#>  3
#> [ CPUf32{1} ] 
```
