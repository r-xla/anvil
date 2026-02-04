# Primitive Transpose

Transposes a tensor according to a permutation.

## Usage

``` r
nvl_transpose(operand, permutation)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- permutation:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimension permutation.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)

## Shapes

Output shape is `shape(operand)[permutation]`.

## StableHLO

Calls
[`stablehlo::hlo_transpose()`](https://r-xla.github.io/stablehlo/reference/hlo_transpose.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(matrix(1:6, nrow = 2))
  nvl_transpose(x, permutation = c(2L, 1L))
})
#> AnvilTensor
#>  1 2
#>  3 4
#>  5 6
#> [ CPUi32{3,2} ] 
```
