# Primitive Dynamic Slice

Extracts a dynamically positioned slice from a tensor. The start
position is specified at runtime via tensor indices.

## Usage

``` r
nvl_dynamic_slice(operand, ..., slice_sizes)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- ...:

  Scalar tensor start indices (1-based), one per dimension.

- slice_sizes:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Size of the slice in each dimension.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)

## Out Of Bounds Behavior

If the slice would extend beyond the bounds of the operand tensor, the
start indices are clamped so that the slice fits within the tensor. This
means that out-of-bounds indices will not cause an error, but the
effective start position may differ from the requested one.

For example, slicing a tensor of shape `(10,)` with `start_indices = 8`
and `slice_sizes = 5` will effectively use `start_indices = 5` to keep
the slice within bounds.
