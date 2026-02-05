# Generate RNG State

Creates an initial RNG state from a seed. This state is required by all
random sampling functions and is updated after each call.

## Usage

``` r
nv_rng_state(seed)
```

## Arguments

- seed:

  (`integer(1)`)  
  Seed value.

## Value

[`nv_tensor`](https://r-xla.github.io/anvil/reference/AnvilTensor.md) of
dtype `ui64` and shape `(2)`.

## See also

Other rng:
[`nv_rbinom()`](https://r-xla.github.io/anvil/reference/nv_rbinom.md),
[`nv_rdunif()`](https://r-xla.github.io/anvil/reference/nv_rdunif.md),
[`nv_rnorm()`](https://r-xla.github.io/anvil/reference/nv_rnorm.md),
[`nv_runif()`](https://r-xla.github.io/anvil/reference/nv_runif.md)

## Examples

``` r
jit_eval({
  state <- nv_rng_state(42L)
  state
})
#> AnvilTensor
#>  42
#>   0
#> [ CPUui64{2} ] 
```
