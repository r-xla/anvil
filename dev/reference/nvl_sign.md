# Primitive Sign

Element-wise sign.

## Usage

``` r
nvl_sign(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish value of data type signed integer or floating-point.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`stablehlo::hlo_sign()`](https://r-xla.github.io/stablehlo/reference/hlo_sign.html).

## See also

[`nv_sign()`](https://r-xla.github.io/anvil/dev/reference/nv_sign.md),
[`sign()`](https://rdrr.io/r/base/sign.html)

## Examples

``` r
jit_eval({
  x <- nv_array(c(-3, 0, 5))
  nvl_sign(x)
})
#> AnvilArray
#>  -1
#>   0
#>   1
#> [ CPUf32{3} ] 
```
