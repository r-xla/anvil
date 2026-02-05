# Primitive Division

Divides two tensors element-wise.

## Usage

``` r
nvl_div(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish values of data type integer, unsigned integer, or
  floating-point. Must have the same shape.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the inputs. It is ambiguous if both
inputs are ambiguous.

## StableHLO

Lowers to
[`stablehlo::hlo_divide()`](https://r-xla.github.io/stablehlo/reference/hlo_divide.html).

## See also

[`nv_div()`](https://r-xla.github.io/anvil/reference/nv_div.md), `/`

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(10, 20, 30))
  y <- nv_tensor(c(2, 5, 10))
  nvl_div(x, y)
})
#> AnvilTensor
#>  5
#>  4
#>  3
#> [ CPUf32{3} ] 
```
