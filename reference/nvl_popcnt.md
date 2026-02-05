# Primitive Population Count

Element-wise population count (number of set bits).

## Usage

``` r
nvl_popcnt(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of data type integer or unsigned integer.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_popcnt()`](https://r-xla.github.io/stablehlo/reference/hlo_popcnt.html).

## See also

[`nv_popcnt()`](https://r-xla.github.io/anvil/reference/nv_popcnt.md)

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(7L, 3L, 15L))
  nvl_popcnt(x)
})
#> AnvilTensor
#>  3
#>  2
#>  4
#> [ CPUi32{3} ] 
```
