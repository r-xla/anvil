# Primitive Cosine

Element-wise cosine.

## Usage

``` r
nvl_cosine(operand)
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
[`stablehlo::hlo_cosine()`](https://r-xla.github.io/stablehlo/reference/hlo_cosine.html).

## See also

[`nv_cosine()`](https://r-xla.github.io/anvil/dev/reference/nv_cosine.md),
[`cos()`](https://rdrr.io/r/base/Trig.html)

## Examples

``` r
jit_eval({
  x <- nv_array(c(0, pi / 2, pi))
  nvl_cosine(x)
})
#> AnvilArray
#>   1.0000e+00
#>  -4.3711e-08
#>  -1.0000e+00
#> [ CPUf32{3} ] 
```
