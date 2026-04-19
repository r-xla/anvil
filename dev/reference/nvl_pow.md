# Primitive Power

Raises lhs to the power of rhs element-wise.

## Usage

``` r
nvl_pow(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish values of data type integer, unsigned integer, or
  floating-point. Must have the same shape.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the inputs. It is ambiguous if both
inputs are ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`stablehlo::hlo_power()`](https://r-xla.github.io/stablehlo/reference/hlo_power.html).

## See also

[`nv_pow()`](https://r-xla.github.io/anvil/dev/reference/nv_pow.md), `^`

## Examples

``` r
jit_eval({
  x <- nv_array(c(2, 3, 4))
  y <- nv_array(c(3, 2, 1))
  nvl_pow(x, y)
})
#> AnvilArray
#>  8
#>  9
#>  4
#> [ CPUf32{3} ] 
```
