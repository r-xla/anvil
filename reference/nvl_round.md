# Primitive Round

Element-wise rounding.

## Usage

``` r
nvl_round(operand, method = "nearest_even")
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
  of floating-point type)  
  Operand.

- method:

  (`character(1)`)  
  Rounding method ("nearest_even" or "afz").

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)

## Shapes

Operand can have any shape. The output has the same shape.

## StableHLO

Calls
[`stablehlo::hlo_round_nearest_even()`](https://r-xla.github.io/stablehlo/reference/hlo_round_nearest_even.html)
or
[`stablehlo::hlo_round_nearest_afz()`](https://r-xla.github.io/stablehlo/reference/hlo_round_nearest_afz.html)
depending on the `method` parameter.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1.4, 2.5, 3.6))
  nvl_round(x)
})
#> AnvilTensor
#>  1
#>  2
#>  4
#> [ CPUf32{3} ] 
```
