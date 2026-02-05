# Primitive RNG Bit Generator

Generates pseudo-random numbers using the specified algorithm and
returns the updated RNG state together with the generated values.

## Usage

``` r
nvl_rng_bit_generator(initial_state, rng_algorithm = "THREE_FRY", dtype, shape)
```

## Arguments

- initial_state:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  RNG state (`ui64[2]`).

- rng_algorithm:

  (`character(1)`)  
  RNG algorithm name. Default is `"THREE_FRY"`.

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type of the generated random values.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

## Value

`list` of two
[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
values:  
The first element is the updated RNG state with the same dtype and shape
as `initial_state`. The second element is a tensor of random values with
the given `dtype` and `shape`.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to
[`stablehlo::hlo_rng_bit_generator()`](https://r-xla.github.io/stablehlo/reference/hlo_rng_bit_generator.html).

## See also

[`nv_runif()`](https://r-xla.github.io/anvil/reference/nv_runif.md),
[`nv_rnorm()`](https://r-xla.github.io/anvil/reference/nv_rnorm.md)

## Examples

``` r
jit_eval({
  state <- nv_tensor(c(0L, 0L), dtype = "ui64")
  nvl_rng_bit_generator(state, dtype = "f32", shape = c(3, 2))
})
#> [[1]]
#> AnvilTensor
#>  0
#>  3
#> [ CPUui64{2} ] 
#> 
#> [[2]]
#> AnvilTensor
#>  1.7973e+09 2.5791e+09
#>  1.3515e+09 3.2358e+09
#>  1.6886e+09 4.2293e+09
#> [ CPUui32{3,2} ] 
#> 
```
