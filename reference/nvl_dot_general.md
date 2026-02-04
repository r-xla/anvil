# Primitive Dot General

General dot product of two tensors.

## Usage

``` r
nvl_dot_general(lhs, rhs, contracting_dims, batching_dims)
```

## Arguments

- lhs, rhs:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Left and right operand.

- contracting_dims:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  Dimensions to contract.

- batching_dims:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  Batch dimensions.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)

## Shapes

Contracting dimensions in `lhs` and `rhs` must have matching sizes.
Batching dimensions must also have matching sizes. The output shape is
the batching dimensions followed by the remaining (non-contracted,
non-batched) dimensions of `lhs`, then `rhs`.

## StableHLO

Calls
[`stablehlo::hlo_dot_general()`](https://r-xla.github.io/stablehlo/reference/hlo_dot_general.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(matrix(1:6, nrow = 2))
  y <- nv_tensor(matrix(1:6, nrow = 3))
  nvl_dot_general(x, y,
    contracting_dims = list(2L, 1L),
    batching_dims = list(integer(0), integer(0))
  )
})
#> AnvilTensor
#>  22 49
#>  28 64
#> [ CPUi32{2,2} ] 
```
