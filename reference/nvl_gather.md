# Primitive Gather

Gathers slices from the `operand` tensor at positions specified by
`start_indices`. Each index vector in `start_indices` identifies a
starting position in `operand`, and a slice of size `slice_sizes` is
extracted from that position. The gathered slices are assembled into the
output tensor.

This is the inverse of
[`nvl_scatter()`](https://r-xla.github.io/anvil/reference/nvl_scatter.md):
gather reads slices from a tensor at given indices, while scatter writes
slices into a tensor at given indices.

## Usage

``` r
nvl_gather(
  operand,
  start_indices,
  slice_sizes,
  offset_dims,
  collapsed_slice_dims,
  operand_batching_dims,
  start_indices_batching_dims,
  start_index_map,
  index_vector_dim,
  indices_are_sorted = FALSE,
  unique_indices = FALSE
)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of any data type.

- start_indices:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
  of integer type)  
  Tensor of starting indices. Contains index vectors that map to
  positions in `operand` via `start_index_map`. The dimension specified
  by `index_vector_dim` holds the index vectors.

- slice_sizes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Size of the slice to gather from `operand` in each dimension. Must
  have length equal to `ndims(operand)`.

- offset_dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions in the output that correspond to the non-collapsed slice
  dimensions of `operand`.

- collapsed_slice_dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions of `operand` that are collapsed (removed) from the slice.
  The corresponding entries in `slice_sizes` must be `1`. Together with
  `offset_dims` and `operand_batching_dims`, these must account for all
  dimensions of `operand`.

- operand_batching_dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions of `operand` that are batch dimensions. Use `integer(0)`
  when there are no batch dimensions.

- start_indices_batching_dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions of `start_indices` that correspond to batch dimensions.
  Must have the same length as `operand_batching_dims`.

- start_index_map:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Maps each component of the index vector to an `operand` dimension. For
  example, `start_index_map = c(1L)` means each index vector indexes
  into the first dimension of `operand`.

- index_vector_dim:

  (`integer(1)`)  
  Dimension of `start_indices` that contains the index vectors. If set
  to `ndims(start_indices) + 1`, each scalar element of `start_indices`
  is treated as a length-1 index vector.

- indices_are_sorted:

  (`logical(1)`)  
  Whether indices are guaranteed to be sorted. Setting to `TRUE` may
  improve performance but produces undefined behavior if the indices are
  not actually sorted. Default `FALSE`.

- unique_indices:

  (`logical(1)`)  
  Whether indices are guaranteed to be unique (no duplicates). Setting
  to `TRUE` may improve performance but produces undefined behavior if
  the indices are not actually unique. Default `FALSE`.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same data type as `operand`. The output shape is composed of the
offset dimensions (from the slice) and the remaining dimensions from
`start_indices`. See the underluing stableHLO function for more details.

## Out Of Bounds Behavior

Start indices are clamped before the slice is extracted:
`clamp(1, start_index, nv_shape(operand) - slice_sizes + 1)`. This means
that out-of-bounds indices will not cause an error, but the effective
start position may differ from the requested one.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_gather()`](https://r-xla.github.io/stablehlo/reference/hlo_gather.html).

## See also

[`nvl_scatter()`](https://r-xla.github.io/anvil/reference/nvl_scatter.md),
[`nv_subset()`](https://r-xla.github.io/anvil/reference/nv_subset.md),
[`nv_subset_assign()`](https://r-xla.github.io/anvil/reference/nv_subset_assign.md),
`[`, `[<-`

## Examples

``` r
# Gather rows 1 and 3 from a 3x3 matrix
jit_eval({
  operand <- nv_tensor(matrix(1:9, nrow = 3))
  indices <- nv_tensor(matrix(c(1L, 3L), ncol = 1))
  nvl_gather(
    operand, indices,
    slice_sizes = c(1L, 3L),
    offset_dims = 2L,
    collapsed_slice_dims = 1L,
    operand_batching_dims = integer(0),
    start_indices_batching_dims = integer(0),
    start_index_map = 1L,
    index_vector_dim = 2L
  )
})
#> AnvilTensor
#>  1 4 7
#>  3 6 9
#> [ CPUi32{2,3} ] 
```
