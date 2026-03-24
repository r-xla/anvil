# Primitive Reverse

Reverses the order of elements along specified dimensions.

## Usage

``` r
nvl_reverse(operand, dims)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions to reverse (1-indexed).

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same data type and shape as `operand`. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_reverse()`](https://r-xla.github.io/stablehlo/reference/hlo_reverse.html).

## See also

[`nv_reverse()`](https://r-xla.github.io/anvil/dev/reference/nv_reverse.md)

## Examples

``` r
jit_eval({
  x <- nv_array(c(1, 2, 3, 4, 5))
  nvl_reverse(x, dims = 1L)
})
#> AnvilArray
#>  5
#>  4
#>  3
#>  2
#>  1
#> [ CPUf32{5} ] 
```
