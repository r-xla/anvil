# Primitive Absolute Value

Element-wise absolute value. Is the same as
[`nv_abs()`](https://r-xla.github.io/anvil/reference/nv_abs.md). You can
also use [`abs()`](https://rdrr.io/r/base/MathFun.html).

## Usage

``` r
nvl_abs(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of data type signed integer or floating-point.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_abs()`](https://r-xla.github.io/stablehlo/reference/hlo_abs.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(-1, 2, -3))
  nvl_abs(x)
})
#> AnvilTensor
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
