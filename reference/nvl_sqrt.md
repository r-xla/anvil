# Primitive Square Root

Element-wise square root.

## Usage

``` r
nvl_sqrt(operand)
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
[`stablehlo::hlo_sqrt()`](https://r-xla.github.io/stablehlo/reference/hlo_sqrt.html).

## See also

[`nv_sqrt()`](https://r-xla.github.io/anvil/reference/nv_sqrt.md),
[`sqrt()`](https://rdrr.io/r/base/MathFun.html)

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 4, 9))
  nvl_sqrt(x)
})
#> AnvilTensor
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
