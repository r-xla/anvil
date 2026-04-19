# Primitive Tangent

Element-wise tangent.

## Usage

``` r
nvl_tan(operand)
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

- `reverse`

## StableHLO

Lowers to
[`stablehlo::hlo_tan()`](https://r-xla.github.io/stablehlo/reference/hlo_tan.html).

## See also

[`nv_tan()`](https://r-xla.github.io/anvil/dev/reference/nv_tan.md),
[`tan()`](https://rdrr.io/r/base/Trig.html)

## Examples

``` r
jit_eval({
  x <- nv_array(c(0, 0.5, 1))
  nvl_tan(x)
})
#> AnvilArray
#>  0.0000
#>  0.5463
#>  1.5574
#> [ CPUf32{3} ] 
```
