# Random Normal Numbers

generate random normal numbers

## Usage

``` r
nv_rnorm(initial_state, dtype, shape_out, mu = 0, sigma = 1)
```

## Arguments

- initial_state:

  state seed

- dtype:

  output dtype either "f32" or "f64"

- shape_out:

  output shape

- mu:

  scalar: expected value

- sigma:

  scalar: standard deviation \#' @section Covariance: To implement a
  covariance structure use cholesky decomposition
