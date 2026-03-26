# Primitive Less Equal

Element-wise less than or equal comparison.

## Usage

``` r
nvl_le(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish values of any data type. Must have the same shape.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape as the inputs and boolean data type. It is ambiguous
if both inputs are ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`stablehlo::hlo_compare()`](https://r-xla.github.io/stablehlo/reference/hlo_compare.html)
with `comparison_direction = "LE"`.

## See also

[`nv_le()`](https://r-xla.github.io/anvil/dev/reference/nv_le.md), `<=`

## Examples

``` r
jit_eval({
  x <- nv_array(c(1, 2, 3))
  y <- nv_array(c(3, 2, 1))
  nvl_le(x, y)
})
#> AnvilArray
#>  1
#>  1
#>  0
#> [ CPUbool{3} ] 
```
