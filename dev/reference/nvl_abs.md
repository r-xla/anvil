# Primitive Absolute Value

Element-wise absolute value.

## Usage

``` r
nvl_abs(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish value of data type signed integer or floating-point.

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
[`stablehlo::hlo_abs()`](https://r-xla.github.io/stablehlo/reference/hlo_abs.html).

## See also

[`nv_abs()`](https://r-xla.github.io/anvil/dev/reference/nv_abs.md),
[`abs()`](https://rdrr.io/r/base/MathFun.html)

## Examples

``` r
jit_eval({
  x <- nv_array(c(-1, 2, -3))
  nvl_abs(x)
})
#> AnvilArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
