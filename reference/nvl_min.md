# Primitive Minimum

Element-wise minimum of two tensors. For a more user-friendly interface,
see [`nv_min()`](https://r-xla.github.io/anvil/reference/nv_min.md).

## Usage

``` r
nvl_min(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish values of any data type. Must have the same shape.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the inputs. It is ambiguous if both
inputs are ambiguous.

## StableHLO

Lowers to
[`stablehlo::hlo_minimum()`](https://r-xla.github.io/stablehlo/reference/hlo_minimum.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 5, 3))
  y <- nv_tensor(c(4, 2, 6))
  nvl_min(x, y)
})
#> AnvilTensor
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
