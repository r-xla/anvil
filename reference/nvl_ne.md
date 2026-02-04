# Primitive Not Equal

Element-wise inequality comparison. For a more user-friendly interface,
see [`nv_ne()`](https://r-xla.github.io/anvil/reference/nv_ne.md), or
use the `!=` operator.

## Usage

``` r
nvl_ne(lhs, rhs)
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
with `comparison_direction = "NE"`.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 2, 3))
  y <- nv_tensor(c(1, 3, 2))
  nvl_ne(x, y)
})
#> AnvilTensor
#>  0
#>  1
#>  1
#> [ CPUi1{3} ] 
```
