# Primitive Scatter

Produces a result tensor equal to the input tensor except that slices
specified by scatter_indices are updated with values from the update
tensor.

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
  Input tensor to scatter into.

- scatter_indices:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
  of integer type)  
  Indices tensor.

- update:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Update values tensor.

- update_window_dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Update window dimensions.

- inserted_window_dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Inserted window dimensions.

- input_batching_dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Input batching dimensions.

- scatter_indices_batching_dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Scatter indices batching dimensions.

- scatter_dims_to_operand_dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Mapping from scatter indices to operand dimensions.

- index_vector_dim:

  (`integer(1)`)  
  Dimension in scatter_indices containing the index vectors.

- indices_are_sorted:

  (`logical(1)`)  
  Whether indices are sorted.

- unique_indices:

  (`logical(1)`)  
  Whether indices are unique.

- update_computation:

  (`function`)  
  Binary function to combine existing and update values.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)

## Shapes

Output has the same shape as `input`. See
[`stablehlo::hlo_scatter()`](https://r-xla.github.io/stablehlo/reference/hlo_scatter.html)
for detailed dimension constraints on `scatter_indices`, `update`, and
the dimension mapping parameters.

## StableHLO

Calls
[`stablehlo::hlo_scatter()`](https://r-xla.github.io/stablehlo/reference/hlo_scatter.html).

## Examples

``` r
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
