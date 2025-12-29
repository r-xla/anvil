# Sample from a Normal Distribution

Sample from a normal distribution with mean \\\mu\\ and standard
deviation \\\sigma\\.

## Usage

``` r
nv_rnorm(shape, initial_state, dtype = "f32", mu = 0, sigma = 1)
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- initial_state:

  (\[`ui64[2]`\]\[tensorish\])  
  RNG state.

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

- mu:

  (`numeric(1)`)  
  Expected value.

- sigma:

  (`numeric(1)`)  
  Standard deviation.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`tensorish`](tensorish.md))  
List of two tensors: the new RNG state and the generated random numbers.

## Covariance

To implement a covariance structure use cholesky decomposition.
