# Primitive Maximum

Element-wise maximum of two tensors.

## Usage

``` r
nvl_max(lhs, rhs)
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
[`stablehlo::hlo_maximum()`](https://r-xla.github.io/stablehlo/reference/hlo_maximum.html).

## See also

[`nv_max()`](https://r-xla.github.io/anvil/reference/nv_max.md)

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 5, 3))
  y <- nv_tensor(c(4, 2, 6))
  nvl_max(x, y)
})
#> AnvilTensor
#>  4
#>  5
#>  6
#> [ CPUf32{3} ] 
```
