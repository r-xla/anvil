# Primitive Minimum

Element-wise minimum of two arrays.

## Usage

``` r
nvl_min(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish values of any data type. Must have the same shape.

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
[`stablehlo::hlo_minimum()`](https://r-xla.github.io/stablehlo/reference/hlo_minimum.html).

## See also

[`nv_min()`](https://r-xla.github.io/anvil/dev/reference/nv_min.md)

## Examples

``` r
jit_eval({
  x <- nv_array(c(1, 5, 3))
  y <- nv_array(c(4, 2, 6))
  nvl_min(x, y)
})
#> AnvilArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
