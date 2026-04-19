# Primitive Division

Divides two arrays element-wise.

## Usage

``` r
nvl_div(lhs, rhs)
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
[`stablehlo::hlo_divide()`](https://r-xla.github.io/stablehlo/reference/hlo_divide.html).

## See also

[`nv_div()`](https://r-xla.github.io/anvil/dev/reference/nv_div.md), `/`

## Examples

``` r
jit_eval({
  x <- nv_array(c(10, 20, 30))
  y <- nv_array(c(2, 5, 10))
  nvl_div(x, y)
})
#> AnvilArray
#>  5
#>  4
#>  3
#> [ CPUf32{3} ] 
```
