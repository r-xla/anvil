# Primitive Is Finite

Element-wise check if values are finite (not Inf, -Inf, or NaN).

## Usage

``` r
nvl_is_finite(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish value of data type floating-point.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape as the input and boolean data type. It is ambiguous
if the input is ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_is_finite()`](https://r-xla.github.io/stablehlo/reference/hlo_is_finite.html).

## See also

[`nv_is_finite()`](https://r-xla.github.io/anvil/dev/reference/nv_is_finite.md)

## Examples

``` r
jit_eval({
  x <- nv_array(c(1, Inf, NaN, -Inf, 0))
  nvl_is_finite(x)
})
#> AnvilArray
#>  1
#>  0
#>  0
#>  0
#>  1
#> [ CPUbool{5} ] 
```
