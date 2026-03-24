# Primitive Ceiling

Element-wise ceiling.

## Usage

``` r
nvl_ceil(operand)
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
[`stablehlo::hlo_ceil()`](https://r-xla.github.io/stablehlo/reference/hlo_ceil.html).

## See also

[`nv_ceil()`](https://r-xla.github.io/anvil/dev/reference/nv_ceil.md),
[`ceiling()`](https://rdrr.io/r/base/Round.html)

## Examples

``` r
jit_eval({
  x <- nv_array(c(1.2, 2.7, -1.5))
  nvl_ceil(x)
})
#> AnvilArray
#>   2
#>   3
#>  -1
#> [ CPUf32{3} ] 
```
