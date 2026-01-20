# Primitive Dynamic Update Slice

Updates a dynamically positioned slice in a tensor. The start position
is specified at runtime via tensor indices.

## Usage

``` r
nvl_dynamic_update_slice(operand, update, ...)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- update:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  The values to write at the specified position.

- ...:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Scalar tensor start indices (1-based), one per dimension.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)

## Out Of Bounds Behavior

If the update slice would extend beyond the bounds of the operand
tensor, the start indices are clamped so that the update fits within the
tensor. This means that out-of-bounds indices will not cause an error,
but the effective start position may differ from the requested one.
