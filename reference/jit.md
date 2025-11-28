# JIT compile a function

Convert a function to a JIT compiled function.

## Usage

``` r
jit(
  f,
  static = character(),
  device = NULL,
  cache_size = 100L,
  donate = character()
)
```

## Arguments

- f:

  (`function`)  
  Function to compile.

- static:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Which parameters of `f` are static.

- device:

  (`NULL` \| `character(1)` \|
  [`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html))  
  The device to use for the compiled function. The default (`NULL`) uses
  the `PJRT_PLATFORM` environment variable or defaults to "cpu".

- cache_size:

  (`integer(1)`)  
  The size of the cache for the jit-compiled functions.

- donate:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of the arguments whose buffers should be donated. Donated
  buffers can be aliased with outputs of the same type, allowing
  in-place operations and reducing memory usage.

## Value

(`function`)
