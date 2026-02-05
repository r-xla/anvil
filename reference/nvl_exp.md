# Primitive Exponential

Element-wise exponential.

## Usage

``` r
nvl_exp(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of data type floating-point.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_exponential()`](https://r-xla.github.io/stablehlo/reference/hlo_exponential.html).

## See also

[`nv_exp()`](https://r-xla.github.io/anvil/reference/nv_exp.md),
[`exp()`](https://rdrr.io/r/base/Log.html)

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(0, 1, 2))
  nvl_exp(x)
})
#> AnvilTensor
#>  1.0000
#>  2.7183
#>  7.3891
#> [ CPUf32{3} ] 
```
