# Primitive All Reduction

Performs logical AND along the specified dimensions.

## Usage

``` r
nvl_reduce_all(operand, dims, drop = TRUE)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of boolean data type.

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
Boolean tensor. Never ambiguous. When `drop = TRUE`, the shape is that
of `operand` with `dims` removed. When `drop = FALSE`, the shape is that
of `operand` with `dims` set to 1.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_reduce()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce.html)
with
[`stablehlo::hlo_and()`](https://r-xla.github.io/stablehlo/reference/hlo_and.html)
as the reducer.

## See also

[`nv_reduce_all()`](https://r-xla.github.io/anvil/reference/nv_reduce_ops.md)

## Examples

``` r
jit_eval({
  x <- nv_tensor(matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2))
  nvl_reduce_all(x, dims = 1L)
})
#> AnvilTensor
#>  0
#>  1
#> [ CPUi1{2} ] 
```
