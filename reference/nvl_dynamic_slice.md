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

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
  of integer type)  
  Scalar start indices (1-based), one per dimension.

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

For example, slicing a tensor of shape `c(10)` with `start_indices = 8`
and `slice_sizes = 5` will effectively use `start_indices = 6` to keep
the slice within bounds.

## Shapes

Each start index in `...` must be a scalar tensor. The number of start
indices must equal `rank(operand)`. `slice_sizes` must satisfy
`slice_sizes <= shape(operand)` per dimension. Output shape is
`slice_sizes`.

## StableHLO

Calls
[`stablehlo::hlo_dynamic_slice()`](https://r-xla.github.io/stablehlo/reference/hlo_dynamic_slice.html).

## Examples

``` r
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
```
