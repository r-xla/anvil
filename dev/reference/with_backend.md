# Temporarily override the active backend

Temporarily override the active backend

## Usage

``` r
with_backend(backend, code)
```

## Arguments

- backend:

  (`character(1)`)  
  Backend to use (`"xla"` or `"quickr"`).

- code:

  Expression to evaluate with the overridden backend.

## Value

The result of evaluating `code`.
