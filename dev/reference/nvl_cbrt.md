# Primitive Cube Root

Element-wise cube root.

## Usage

``` r
nvl_cbrt(operand)
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

- `reverse`

## StableHLO

Lowers to
[`stablehlo::hlo_cbrt()`](https://r-xla.github.io/stablehlo/reference/hlo_cbrt.html).

## See also

[`nv_cbrt()`](https://r-xla.github.io/anvil/dev/reference/nv_cbrt.md)

## Examples

``` r
jit_eval({
  x <- nv_array(c(1, 8, 27))
  nvl_cbrt(x)
})
#> AnvilArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
