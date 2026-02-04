# Primitive RNG Bit Generator

Generates random bits using the specified algorithm.

## Usage

``` r
nvl_rng_bit_generator(initial_state, rng_algorithm = "THREE_FRY", dtype, shape)
```

## Arguments

- initial_state:

  (1-d
  [`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md) of
  type `ui64`)  
  RNG state tensor.

- rng_algorithm:

  (`character(1)`)  
  Algorithm name (default "THREE_FRY").

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Output shape.

## Value

List of new state and random tensor.

## Shapes

`initial_state` must be 1-d. Returns a list with the updated state (same
shape as `initial_state`) and a random tensor with the specified
`shape`.

## StableHLO

Calls
[`stablehlo::hlo_rng_bit_generator()`](https://r-xla.github.io/stablehlo/reference/hlo_rng_bit_generator.html).

## Examples

``` r
jit_eval({
  state <- nv_tensor(c(0L, 0L), dtype = "ui64")
  nvl_rng_bit_generator(state, dtype = "f32", shape = c(3))
})
#> [[1]]
#> AnvilTensor
#>  0
#>  2
#> [ CPUui64{2} ] 
#> 
#> [[2]]
#> AnvilTensor
#>  1.7973e+09
#>  2.5791e+09
#>  1.3515e+09
#> [ CPUui32{3} ] 
#> 
```
