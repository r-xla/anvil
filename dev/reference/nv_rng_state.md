# Generate RNG State

Creates an initial RNG state from a seed. This state is required by all
random sampling functions and is updated after each call.

## Usage

``` r
nv_rng_state(seed, device = default_device())
```

## Arguments

- seed:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Scalar `i32` seed value.

- device:

  ( `character(1)` \| `PJRTDevice` \|
  [`quickr_device`](https://r-xla.github.io/anvl/dev/reference/quickr_device.md)
  \| `NULL`)  
  Device for data to live on.

## Value

[`nv_array`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md) of
dtype `ui64` and shape `(2)`.

## See also

Other rng:
[`nv_rbinom()`](https://r-xla.github.io/anvl/dev/reference/nv_rbinom.md),
[`nv_rdunif()`](https://r-xla.github.io/anvl/dev/reference/nv_rdunif.md),
[`nv_rnorm()`](https://r-xla.github.io/anvl/dev/reference/nv_rnorm.md),
[`nv_runif()`](https://r-xla.github.io/anvl/dev/reference/nv_runif.md)

## Examples

``` r
state <- nv_rng_state(42L)
state
#> AnvlArray
#>  42
#>   0
#> [ CPUui64{2} ] 
```
