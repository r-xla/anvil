# Primitive Power

Raises lhs to the power of rhs element-wise. For a more user-friendly
interface, see
[`nv_pow()`](https://r-xla.github.io/anvil/reference/nv_pow.md), or use
the `^` operator.

## Usage

``` r
nvl_pow(lhs, rhs)
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
[`stablehlo::hlo_power()`](https://r-xla.github.io/stablehlo/reference/hlo_power.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(2, 3, 4))
  y <- nv_tensor(c(3, 2, 1))
  nvl_pow(x, y)
})
#> AnvilTensor
#>  8
#>  9
#>  4
#> [ CPUf32{3} ] 
```
