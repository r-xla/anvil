# Primitive Less Than

Element-wise less than comparison. For a more user-friendly interface,
see [`nv_lt()`](https://r-xla.github.io/anvil/reference/nv_lt.md), or
use the `<` operator.

## Usage

``` r
nvl_lt(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish values of any data type. Must have the same shape.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape as the inputs and boolean data type. It is ambiguous
if both inputs are ambiguous.

## StableHLO

Lowers to
[`stablehlo::hlo_compare()`](https://r-xla.github.io/stablehlo/reference/hlo_compare.html)
with `comparison_direction = "LT"`.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 2, 3))
  y <- nv_tensor(c(3, 2, 1))
  nvl_lt(x, y)
})
#> AnvilTensor
#>  1
#>  0
#>  0
#> [ CPUi1{3} ] 
```
