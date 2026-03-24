# Primitive Square Root

Element-wise square root.

## Usage

``` r
nvl_sqrt(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish value of data type floating-point.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_sqrt()`](https://r-xla.github.io/stablehlo/reference/hlo_sqrt.html).

## See also

[`nv_sqrt()`](https://r-xla.github.io/anvil/dev/reference/nv_sqrt.md),
[`sqrt()`](https://rdrr.io/r/base/MathFun.html)

## Examples

``` r
jit_eval({
  x <- nv_array(c(1, 4, 9))
  nvl_sqrt(x)
})
#> AnvilArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
