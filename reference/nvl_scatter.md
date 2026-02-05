# Primitive Scatter

Produces a result tensor identical to `input` except that slices at
positions specified by `scatter_indices` are updated with values from
the `update` tensor. When multiple indices point to the same location,
the `update_computation` function determines how to combine the values
(by default the new value replaces the old one).

This is the inverse of
[`nvl_gather()`](https://r-xla.github.io/anvil/reference/nvl_gather.md):
gather reads slices from a tensor at given indices, while scatter writes
slices into a tensor at given indices.

## Usage

``` r
nvl_scatter(
  input,
  scatter_indices,
  update,
  update_window_dims,
  inserted_window_dims,
  input_batching_dims,
  scatter_indices_batching_dims,
  scatter_dims_to_operand_dims,
  index_vector_dim,
  indices_are_sorted = FALSE,
  unique_indices = FALSE,
  update_computation = NULL
)
```

## Arguments

- input:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of any data type. The base tensor to scatter into.

- scatter_indices:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
  of integer type)  
  Tensor of indices. Contains index vectors that map to positions in
  `input` via `scatter_dims_to_operand_dims`. The dimension specified by
  `index_vector_dim` holds the index vectors.

- update:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Update values tensor. Must have the same data type as `input`.

- update_window_dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions of `update` that are window dimensions, i.e. they
  correspond to the slice being written into `input`.

- inserted_window_dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions of `input` whose slices have size 1 and are inserted (not
  present) in the `update` window. Together with `update_window_dims`
  and `input_batching_dims`, these must account for all dimensions of
  `input`.

- input_batching_dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions of `input` that are batch dimensions. Use `integer(0)` when
  there are no batch dimensions.

- scatter_indices_batching_dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions of `scatter_indices` that correspond to batch dimensions.
  Must have the same length as `input_batching_dims`.

- scatter_dims_to_operand_dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Maps each component of the index vector to an `input` dimension. For
  example, `scatter_dims_to_operand_dims = c(1L)` means each index
  vector indexes into the first dimension of `input`.

- index_vector_dim:

  (`integer(1)`)  
  Dimension of `scatter_indices` that contains the index vectors. If set
  to `ndims(scatter_indices) + 1`, each scalar element of
  `scatter_indices` is treated as a length-1 index vector.

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

- update_computation:

  (`function`)  
  Binary function `f(old, new)` that combines the existing value in
  `input` with the value from `update`. The default (`NULL`) uses
  `function(old, new) new`, which replaces the old value.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same data type and shape as `input`. It is ambiguous if `input`
is ambiguous.

## Out Of Bounds Behavior

If a computed result index falls outside the bounds of `input`, the
update for that index is silently ignored.

## Update Order

When multiple indices in `scatter_indices` map to the same element of
`input`, the order in which `update_computation` is applied is
implementation-defined and may vary between plugins ("cpu", "cuda").

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_scatter()`](https://r-xla.github.io/stablehlo/reference/hlo_scatter.html).

## See also

[`nvl_gather()`](https://r-xla.github.io/anvil/reference/nvl_gather.md),
[`nv_subset()`](https://r-xla.github.io/anvil/reference/nv_subset.md),
[`nv_subset_assign()`](https://r-xla.github.io/anvil/reference/nv_subset_assign.md),
`[`, `[<-`

## Examples

``` r
# Scatter values 10 and 30 into positions 1 and 3 of a zero vector
jit_eval({
  input <- nv_tensor(c(0, 0, 0, 0, 0))
  indices <- nv_tensor(matrix(c(1L, 3L), ncol = 1))
  updates <- nv_tensor(c(10, 30))
  nvl_scatter(
    input, indices, updates,
    update_window_dims = integer(0),
    inserted_window_dims = 1L,
    input_batching_dims = integer(0),
    scatter_indices_batching_dims = integer(0),
    scatter_dims_to_operand_dims = 1L,
    index_vector_dim = 2L
  )
})
#> AnvilTensor
#>  10
#>   0
#>  30
#>   0
#>   0
#> [ CPUf32{5} ] 
```
