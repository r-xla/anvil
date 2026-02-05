# Static Slice

Extracts a slice from a tensor using static (compile-time) indices. For
dynamic indexing, use
[`nv_subset()`](https://r-xla.github.io/anvil/reference/nv_subset.md)
instead.

## Usage

``` r
nv_static_slice(operand, start_indices, limit_indices, strides)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- start_indices:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Start indices (inclusive), one per dimension.

- limit_indices:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  End indices (inclusive), one per dimension.

- strides:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Step sizes, one per dimension. A stride of 1 selects every element.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same data type as `operand`.

## See also

[`nv_subset()`](https://r-xla.github.io/anvil/reference/nv_subset.md),
[`nvl_static_slice()`](https://r-xla.github.io/anvil/reference/nvl_static_slice.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(1:10)
  nv_static_slice(x, start_indices = 2L, limit_indices = 5L, strides = 1L)
})
#> AnvilTensor
#>  2
#>  3
#>  4
#>  5
#> [ CPUi32{4} ] 
```
