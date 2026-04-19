# Primitive Not

Element-wise logical NOT.

## Usage

``` r
nvl_not(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish value of data type boolean, integer, or unsigned integer.

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
[`stablehlo::hlo_not()`](https://r-xla.github.io/stablehlo/reference/hlo_not.html).

## See also

[`nv_not()`](https://r-xla.github.io/anvil/dev/reference/nv_not.md)

## Examples

``` r
jit_eval({
  x <- nv_array(c(TRUE, FALSE, TRUE))
  nvl_not(x)
})
#> AnvilArray
#>  0
#>  1
#>  0
#> [ CPUbool{3} ] 
```
