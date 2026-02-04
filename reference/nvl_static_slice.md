# Primitive Static Slice

Extracts a slice from a tensor using static (compile-time) indices. For
dynamic indices, use
[`nvl_dynamic_slice()`](https://r-xla.github.io/anvil/reference/nvl_dynamic_slice.md).

## Usage

``` r
nvl_static_slice(operand, start_indices, limit_indices, strides)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- start_indices:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Start indices (1-based).

- limit_indices:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  End indices (exclusive).

- strides:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Step sizes.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)

## Shapes

`start_indices`, `limit_indices`, and `strides` must each have length
equal to `rank(operand)`. Output shape is
`ceiling((limit_indices - start_indices) / strides)`.

## StableHLO

Calls
[`stablehlo::hlo_slice()`](https://r-xla.github.io/stablehlo/reference/hlo_slice.html).

## Examples

``` r
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
```
