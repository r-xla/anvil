# Sample from a Normal Distribution

Samples from a normal distribution with mean \\\mu\\ and standard
deviation \\\sigma\\ using the Box-Muller transform.

## Usage

``` r
nv_rnorm(shape, initial_state, dtype = "f32", mu = 0, sigma = 1)
```

## Arguments

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- initial_state:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  RNG state (`ui64[2]`).

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

- mu:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Mean.

- sigma:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Standard deviation. Must be positive, otherwise results are invalid.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
List of two elements: the updated RNG state and the sampled values.

## Covariance

To implement a covariance structure use Cholesky decomposition.

## See also

Other rng:
[`nv_rbinom()`](https://r-xla.github.io/anvil/reference/nv_rbinom.md),
[`nv_rdunif()`](https://r-xla.github.io/anvil/reference/nv_rdunif.md),
[`nv_rng_state()`](https://r-xla.github.io/anvil/reference/nv_rng_state.md),
[`nv_runif()`](https://r-xla.github.io/anvil/reference/nv_runif.md)

## Examples

``` r
jit_eval({
  state <- nv_rng_state(42L)
  result <- nv_rnorm(c(2, 3), state)
  result[[2]]
})
#> AnvilTensor
#>  -0.0675  0.9489  1.9457
#>  -0.5255  1.2002  0.0008
#> [ CPUf32{2,3} ] 
```
