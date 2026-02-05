# Primitive Dynamic Slice

Extracts a slice from a tensor whose start position is determined at
runtime via tensor-valued indices. The slice shape (`slice_sizes`) is a
fixed R integer vector.

Use
[`nvl_static_slice()`](https://r-xla.github.io/anvil/reference/nvl_static_slice.md)
instead when all indices are known at compile time and you need stride
support.

## Usage

``` r
nvl_dynamic_slice(operand, ..., slice_sizes)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of any data type.

- ...:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
  of integer type)  
  Scalar start indices, one per dimension. Each must be a scalar tensor.
  Pass one scalar per dimension of `operand`.

- slice_sizes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Size of the slice in each dimension. Must have length equal to
  `ndims(operand)` and satisfy `1 <= slice_sizes <= nv_shape(operand)`
  per dimension.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same data type as the input and shape `slice_sizes`. It is
ambiguous if the input is ambiguous.

## Out Of Bounds Behavior

Start indices are clamped before the slice is extracted:
`adjusted_start_indices = clamp(1, start_indices, nv_shape(operand) - slice_sizes + 1)`.
This means that out-of-bounds indices will not cause an error, but the
effective start position may differ from the requested one.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_dynamic_slice()`](https://r-xla.github.io/stablehlo/reference/hlo_dynamic_slice.html).

## See also

[`nvl_static_slice()`](https://r-xla.github.io/anvil/reference/nvl_static_slice.md),
[`nvl_dynamic_update_slice()`](https://r-xla.github.io/anvil/reference/nvl_dynamic_update_slice.md),
[`nvl_scatter()`](https://r-xla.github.io/anvil/reference/nvl_scatter.md),
[`nvl_gather()`](https://r-xla.github.io/anvil/reference/nvl_gather.md),
[`nv_subset()`](https://r-xla.github.io/anvil/reference/nv_subset.md),
`[`

## Examples

``` r
# 1-D: extract 3 elements starting at position 3
jit_eval({
  x <- nv_tensor(1:10)
  start <- nv_scalar(3L)
  nvl_dynamic_slice(x, start, slice_sizes = 3L)
})
#> AnvilTensor
#>  3
#>  4
#>  5
#> [ CPUi32{3} ] 

# 2-D: extract a 2x2 block from a matrix
jit_eval({
  x <- nv_tensor(matrix(1:12, nrow = 3, ncol = 4))
  row_start <- nv_scalar(2L)
  col_start <- nv_scalar(1L)
  nvl_dynamic_slice(x, row_start, col_start, slice_sizes = c(2L, 2L))
})
#> AnvilTensor
#>  2 5
#>  3 6
#> [ CPUi32{2,2} ] 
```
