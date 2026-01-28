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

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensor of integer type containing indices.

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
