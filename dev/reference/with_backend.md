# Run code with a specific backend

Sets the `anvil.default_backend` option for the duration of the
expression. This affects
[`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md) and data
construction (e.g. via
[`nv_array`](https://r-xla.github.io/anvil/dev/reference/AnvilArray.md)).

## Usage

``` r
with_backend(backend, code)
```

## Arguments

- backend:

  (`character(1)`)  
  Backend to use (`"xla"` or `"quickr"`).

- code:

  An expression to evaluate with the given backend.

## Value

The result of evaluating `code`.
