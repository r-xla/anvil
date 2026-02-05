# Primitive Static Slice

Extracts a slice from a tensor using static (compile-time) indices. All
indices, limits, and strides are fixed R integers.

Use
[`nvl_dynamic_slice()`](https://r-xla.github.io/anvil/reference/nvl_dynamic_slice.md)
instead when the start position must be computed at runtime (e.g.
depends on tensor values).

## Usage

``` r
nvl_static_slice(operand, start_indices, limit_indices, strides)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of any data type.

- start_indices:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Start indices (inclusive), one per dimension. Must satisfy
  `1 <= start_indices <= limit_indices` per dimension.

- limit_indices:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  End indices (inclusive), one per dimension. Must satisfy
  `limit_indices <= nv_shape(operand)` per dimension.

- strides:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Step sizes, one per dimension. Must be `>= 1`. A stride of `1` selects
  every element; a stride of `2` selects every other element, etc.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same data type as the input and shape
`ceiling((limit_indices - start_indices + 1) / strides)`. It is
ambiguous if the input is ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_slice()`](https://r-xla.github.io/stablehlo/reference/hlo_slice.html).

## See also

[`nvl_dynamic_slice()`](https://r-xla.github.io/anvil/reference/nvl_dynamic_slice.md),
[`nvl_scatter()`](https://r-xla.github.io/anvil/reference/nvl_scatter.md),
[`nvl_gather()`](https://r-xla.github.io/anvil/reference/nvl_gather.md),
[`nv_subset()`](https://r-xla.github.io/anvil/reference/nv_subset.md),
`[`

## Examples

``` r
# 1-D: extract elements 2 through 4 (limit is exclusive)
jit_eval({
  x <- nv_tensor(1:10)
  nvl_static_slice(x, start_indices = 2L, limit_indices = 5L, strides = 1L)
})
#> AnvilTensor
#>  2
#>  3
#>  4
#>  5
#> [ CPUi32{4} ] 

# 1-D: every other element using strides
jit_eval({
  x <- nv_tensor(1:10)
  nvl_static_slice(x, start_indices = 1L, limit_indices = 10L, strides = 2L)
})
#> AnvilTensor
#>  1
#>  3
#>  5
#>  7
#>  9
#> [ CPUi32{5} ] 

# 2-D: extract a submatrix (rows 1-2, columns 2-3)
jit_eval({
  x <- nv_tensor(matrix(1:12, nrow = 3, ncol = 4))
  nvl_static_slice(x,
    start_indices = c(1L, 2L),
    limit_indices = c(3L, 4L),
    strides       = c(1L, 1L)
  )
})
#> AnvilTensor
#>   4  7 10
#>   5  8 11
#>   6  9 12
#> [ CPUi32{3,3} ] 
```
