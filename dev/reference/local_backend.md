# Temporarily set the default backend

Sets the `anvil.default_backend` option for the duration of the calling
scope. This affects
[`nv_array()`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md),
[`nv_scalar()`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md),
and [`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md).

## Usage

``` r
local_backend(backend, envir = parent.frame())
```

## Arguments

- backend:

  (`character(1)`)  
  Backend to use (`"xla"` or `"quickr"`).

- envir:

  The environment to scope the change to.

## Value

The previous value of the option (invisibly).
