# Primitive Dynamic Update Slice

Returns a copy of `operand` with a slice replaced by `update` at a
runtime-determined position. This is the write counterpart of
[`nvl_dynamic_slice()`](https://r-xla.github.io/anvil/dev/reference/nvl_dynamic_slice.md):
dynamic slice reads a block from an array, while dynamic update slice
writes a block into an array.

## Usage

``` r
nvl_dynamic_update_slice(operand, update, ...)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- update:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  The values to write at the specified position. Must have the same data
  type and number of dimensions as `operand`, with
  `nv_shape(update) <= nv_shape(operand)` per dimension.

- ...:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)
  of integer type)  
  Scalar start indices, one per dimension of `operand`. Each must be a
  scalar array.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same data type and shape as `operand`. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`stablehlo::hlo_dynamic_update_slice()`](https://r-xla.github.io/stablehlo/reference/hlo_dynamic_update_slice.html).

## Out Of Bounds Behavior

Start indices are clamped before the slice is extracted:
`adjusted_start_indices = clamp(1, start_indices, nv_shape(operand) - slice_sizes + 1)`.
This means that out-of-bounds indices will not cause an error, but the
effective start position may differ from the requested one.

## See also

[`nvl_dynamic_slice()`](https://r-xla.github.io/anvil/dev/reference/nvl_dynamic_slice.md),
[`nvl_scatter()`](https://r-xla.github.io/anvil/dev/reference/nvl_scatter.md),
[`nvl_gather()`](https://r-xla.github.io/anvil/dev/reference/nvl_gather.md),
[`nv_subset_assign()`](https://r-xla.github.io/anvil/dev/reference/nv_subset_assign.md),
`[<-`

## Examples

``` r
# 1-D: overwrite two elements starting at position 2
jit_eval({
  x <- nv_array(1:5)
  update <- nv_array(c(10L, 20L))
  start <- nv_scalar(2L)
  nvl_dynamic_update_slice(x, update, start)
})
#> AnvilArray
#>   1
#>  10
#>  20
#>   4
#>   5
#> [ CPUi32{5} ] 

# 2-D: write a 2x2 block into a 3x4 matrix
jit_eval({
  x <- nv_array(matrix(0L, nrow = 3, ncol = 4))
  update <- nv_array(matrix(c(1L, 2L, 3L, 4L), nrow = 2, ncol = 2))
  row_start <- nv_scalar(2L)
  col_start <- nv_scalar(3L)
  nvl_dynamic_update_slice(x, update, row_start, col_start)
})
#> AnvilArray
#>  0 0 0 0
#>  0 0 1 3
#>  0 0 2 4
#> [ CPUi32{3,4} ] 
```
