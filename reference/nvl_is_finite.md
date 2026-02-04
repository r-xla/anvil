# Primitive Is Finite

Element-wise check if values are finite (not Inf, -Inf, or NaN). Is the
same as
[`nv_is_finite()`](https://r-xla.github.io/anvil/reference/nv_is_finite.md).

## Usage

``` r
nvl_is_finite(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of data type floating-point.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape as the input and boolean data type. It is ambiguous
if the input is ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_is_finite()`](https://r-xla.github.io/stablehlo/reference/hlo_is_finite.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, Inf, NaN, -Inf, 0))
  nvl_is_finite(x)
})
#> AnvilTensor
#>  1
#>  0
#>  0
#>  0
#>  1
#> [ CPUi1{5} ] 
```
