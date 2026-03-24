# Primitive Negation

Negates an array element-wise.

## Usage

``` r
nvl_negate(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish value of data type integer or floating-point.

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
[`stablehlo::hlo_negate()`](https://r-xla.github.io/stablehlo/reference/hlo_negate.html).

## See also

[`nv_negate()`](https://r-xla.github.io/anvil/dev/reference/nv_negate.md),
unary `-`

## Examples

``` r
jit_eval({
  x <- nv_array(c(1, -2, 3))
  nvl_negate(x)
})
#> AnvilArray
#>  -1
#>   2
#>  -3
#> [ CPUf32{3} ] 
```
