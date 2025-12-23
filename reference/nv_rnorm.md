# Random Normal Numbers

generate random normal numbers

## Usage

``` r
nv_rnorm(initial_state, dtype, shape, mu = 0, sigma = 1)
```

## Arguments

- initial_state:

  ([`tensorish`](tensorish.md))  
  Tensor of type `ui64[2]`.

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- mu:

  (`numeric(1)`)  
  Expected value.

- sigma:

  (`numeric(1)`)  
  Standard deviation. \#' @section Covariance: To implement a covariance
  structure use cholesky decomposition.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`tensorish`](tensorish.md))  
List of two tensors: the new RNG state and the generated random numbers.
