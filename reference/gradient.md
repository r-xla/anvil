# Gradient of a function

Compute the gradient of a function using reverse mode automatic
differentiation. Output must be a 0-dimensional tensor.

## Usage

``` r
gradient(f, wrt = NULL)
```

## Arguments

- f:

  (`function`)

- wrt:

  ([`character()`](https://rdrr.io/r/base/character.html)) Names of
  arguments to differentiate with respect to. If `NULL`, compute
  gradients w.r.t. all differentiable arguments.

## Value

(`function`)
