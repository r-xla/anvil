# Primitive Min Reduction

Finds the minimum of tensor elements along the specified dimensions. Is
the same as
[`nv_reduce_min()`](https://r-xla.github.io/anvil/reference/nv_reduce_ops.md).

## Usage

``` r
nvl_reduce_min(operand, dims, drop = TRUE)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of any data type.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions to reduce over.

- drop:

  (`logical(1)`)  
  Whether to drop the reduced dimensions from the output shape. If
  `TRUE`, the reduced dimensions are removed. If `FALSE`, the reduced
  dimensions are set to 1.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same data type as the input. When `drop = TRUE`, the shape is
that of `operand` with `dims` removed. When `drop = FALSE`, the shape is
that of `operand` with `dims` set to 1. It is ambiguous if the input is
ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_reduce()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce.html)
with
[`stablehlo::hlo_minimum()`](https://r-xla.github.io/stablehlo/reference/hlo_minimum.html)
as the reducer.

## Examples

``` r
jit_eval({
  x <- nv_tensor(matrix(1:6, nrow = 2))
  nvl_reduce_min(x, dims = 1L)
})
#> AnvilTensor
#>  1
#>  3
#>  5
#> [ CPUi32{3} ] 
```
