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

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
  of integer type)  
  Scalar start indices (1-based), one per dimension.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)

## Shapes

`update` must have the same rank as `operand`, with
`shape(update) <= shape(operand)` per dimension. Each start index in
`...` must be a scalar tensor. The output has the same shape as
`operand`.

## StableHLO

Calls
[`stablehlo::hlo_dynamic_update_slice()`](https://r-xla.github.io/stablehlo/reference/hlo_dynamic_update_slice.html).

## Out Of Bounds Behavior

If the slice would extend beyond the bounds of the operand tensor, the
start indices are clamped so that the slice fits within the tensor. This
means that out-of-bounds indices will not cause an error, but the
effective start position may differ from the requested one.

For example, slicing a tensor of shape `c(10)` with `start_indices = 8`
and `slice_sizes = 5` will effectively use `start_indices = 6` to keep
the slice within bounds.

## Examples

``` r
jit_eval({
  x <- nv_tensor(1:5)
  update <- nv_tensor(c(10L, 20L))
  start <- nv_scalar(2L)
  nvl_dynamic_update_slice(x, update, start)
})
#> AnvilTensor
#>   1
#>  10
#>  20
#>   4
#>   5
#> [ CPUi32{5} ] 
```
