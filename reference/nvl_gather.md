# Primitive Gather

Gathers slices from the operand at positions specified by start_indices.

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
  Operand.

- start_indices:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
  of integer type)  
  Starting indices for the gather operation.

- slice_sizes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  The sizes of the slices to gather in each dimension.

- offset_dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions of the operand to gather from.

- collapsed_slice_dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions of the slice to gather.

- operand_batching_dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions of the operand to gather from.

- start_indices_batching_dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions of the start_indices to gather from.

- start_index_map:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Mapping from the start_indices to the operand dimensions.

- index_vector_dim:

  (`integer(1)`)  
  Dimension of the index vector.

- indices_are_sorted:

  (`logical(1)`)  
  Whether indices are guaranteed to be sorted.

- unique_indices:

  (`logical(1)`)  
  Whether indices are guaranteed to be unique (no duplicates).

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)

## Shapes

See
[`stablehlo::hlo_gather()`](https://r-xla.github.io/stablehlo/reference/hlo_gather.html)
for detailed dimension constraints on `start_indices`, `slice_sizes`,
and the dimension mapping parameters.

## StableHLO

Calls
[`stablehlo::hlo_gather()`](https://r-xla.github.io/stablehlo/reference/hlo_gather.html).

## Examples

``` r
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
