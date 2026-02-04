# Primitive Reverse

Reverses the order of elements along specified dimensions.

## Usage

``` r
nvl_reverse(operand, dims)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions to reverse (1-indexed).

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)

## Shapes

Output has the same shape as `operand`.

## StableHLO

Calls
[`stablehlo::hlo_reverse()`](https://r-xla.github.io/stablehlo/reference/hlo_reverse.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 2, 3, 4, 5))
  nvl_reverse(x, dims = 1L)
})
#> AnvilTensor
#>  5
#>  4
#>  3
#>  2
#>  1
#> [ CPUf32{5} ] 
```
