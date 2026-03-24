# Primitive Max Reduction

Finds the maximum of array elements along the specified dimensions.

## Usage

``` r
nvl_reduce_max(operand, dims, drop = TRUE)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions to reduce over.

- drop:

  (`logical(1)`)  
  Whether to drop the reduced dimensions from the output shape. If
  `TRUE`, the reduced dimensions are removed. If `FALSE`, the reduced
  dimensions are set to 1.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same data type as the input. When `drop = TRUE`, the shape is
that of `operand` with `dims` removed. When `drop = FALSE`, the shape is
that of `operand` with `dims` set to 1. It is ambiguous if the input is
ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_reduce()`](https://r-xla.github.io/stablehlo/reference/hlo_reduce.html)
with
[`stablehlo::hlo_maximum()`](https://r-xla.github.io/stablehlo/reference/hlo_maximum.html)
as the reducer.

## See also

[`nv_reduce_max()`](https://r-xla.github.io/anvil/dev/reference/nv_reduce_max.md)

## Examples

``` r
jit_eval({
  x <- nv_array(matrix(1:6, nrow = 2))
  nvl_reduce_max(x, dims = 1L)
})
#> AnvilArray
#>  2
#>  4
#>  6
#> [ CPUi32{3} ] 
```
