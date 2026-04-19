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
  backend = NULL,
  device = NULL,
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

  (`NULL` \| `character(1)`)  
  Compilation backend (e.g. `"xla"`, `"quickr"`). The special value
  `"auto"` defers backend selection to call-time. `NULL` (default)
  respects `device` and otherwise falls back to
  [`default_backend()`](https://r-xla.github.io/anvil/dev/reference/default_backend.md).

- device:

  (`NULL` \| `character(1)` \|
  [`nv_device`](https://r-xla.github.io/anvil/dev/reference/nv_device.md)
  \|
  [`device_arg()`](https://r-xla.github.io/anvil/dev/reference/device_arg.md))  
  Target device. When a concrete device is specified, all arrays are
  moved to it.

  The default (`NULL`) infers the device at call time, falling back to
  [`default_device()`](https://r-xla.github.io/anvil/dev/reference/default_device.md).

  In order to use dynamic device selection with the `"auto"` backend
  (e.g. for functions without dynamic inputs such as constant creation),
  set `device = device_arg("<arg>")`.

- ...:

  Backend-specific options. Passing an option that is not supported by
  the selected backend raises an error. See the **XLA JIT arguments**
  and **Quickr JIT arguments** sections below for the options accepted
  by each backend.

## Value

A `JitFunction` (a `function` with the same formals as `f`). The
returned wrapper expects
[`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)
inputs and returns
[`AnvilArray`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)
values.

## Device and Backend selection

There are various ways to specify which device and which backend to use.

**Concrete backend**: In the case where we fix a concrete backend
(backend is not `"auto"`), the device can be inferred or set explicitly.
Setting the device explicitly allows you to enforce that the function
always uses the specified device, e.g. `"cuda:0"`. If the `device`
argument is set, all encountered arrays are copied to it.

If the device is not specified (`NULL`; default) the device will be
inferred from the input arrays and the constants within the program. If
conflicting devices are found, an error is thrown. If no array with a
device is found, we fall back to the default device.

**Auto backend**: When setting `backend = "auto"`, the backend will be
inferred from the array inputs and otherwise fall back to the default
backend. If you want to `jit()` a function without array inputs but make
it work with different devices, set `device = device_arg("<argname>")`
where `<argname>` is the name of the argument specifying the device.
Note that this is only necessary with the `"auto"` backend. When using a
concrete backend, you can just specify the device via a static argument.

## XLA JIT arguments

- `donate` ([`character()`](https://rdrr.io/r/base/character.html),
  default [`character()`](https://rdrr.io/r/base/character.html)): names
  of arguments whose underlying buffers may be donated to (i.e.,
  reused/consumed by) the compiled XLA executable. Donated buffers must
  not be used again by the caller after the call; this can reduce memory
  usage and copies for large inputs. Must not overlap with `static`.

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
