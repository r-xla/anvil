# Primitive Equal

Element-wise equality comparison.

## Usage

``` r
nvl_eq(lhs, rhs)
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

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_compare()`](https://r-xla.github.io/stablehlo/reference/hlo_compare.html)
with `comparison_direction = "EQ"`.

## See also

[`nv_eq()`](https://r-xla.github.io/anvil/dev/reference/nv_eq.md), `==`

## Examples

``` r
jit_eval({
  x <- nv_array(c(1, 2, 3))
  y <- nv_array(c(1, 3, 2))
  nvl_eq(x, y)
})
#> AnvilArray
#>  1
#>  0
#>  0
#> [ CPUbool{3} ] 
```
