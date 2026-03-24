# Primitive Population Count

Element-wise population count (number of set bits).

## Usage

``` r
nvl_popcnt(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish value of data type integer or unsigned integer.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_popcnt()`](https://r-xla.github.io/stablehlo/reference/hlo_popcnt.html).

## See also

[`nv_popcnt()`](https://r-xla.github.io/anvil/dev/reference/nv_popcnt.md)

## Examples

``` r
jit_eval({
  x <- nv_array(c(7L, 3L, 15L))
  nvl_popcnt(x)
})
#> AnvilArray
#>  3
#>  2
#>  4
#> [ CPUi32{3} ] 
```
