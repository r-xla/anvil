# Primitive Hyperbolic Tangent

Element-wise hyperbolic tangent.

## Usage

``` r
nvl_tanh(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish value of data type floating-point.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`stablehlo::hlo_tanh()`](https://r-xla.github.io/stablehlo/reference/hlo_tanh.html).

## See also

[`nv_tanh()`](https://r-xla.github.io/anvil/dev/reference/nv_tanh.md),
[`tanh()`](https://rdrr.io/r/base/Hyperbolic.html)

## Examples

``` r
jit_eval({
  x <- nv_array(c(-1, 0, 1))
  nvl_tanh(x)
})
#> AnvilArray
#>  -0.7616
#>   0.0000
#>   0.7616
#> [ CPUf32{3} ] 
```
