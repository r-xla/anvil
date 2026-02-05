# Primitive Dot General

General dot product of two tensors, supporting contraction over
arbitrary dimensions and batching.

## Usage

``` r
nvl_dot_general(lhs, rhs, contracting_dims, batching_dims)
```

## Arguments

- lhs, rhs:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Left and right operand.

- contracting_dims:

  (`list(integer(), integer())`)  
  A list of two integer vectors specifying which dimensions of `lhs` and
  `rhs` to contract over. The contracted dimensions must have matching
  sizes.

- batching_dims:

  (`list(integer(), integer())`)  
  A list of two integer vectors specifying which dimensions of `lhs` and
  `rhs` are batch dimensions. These must have matching sizes.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
The output shape is the batch dimensions followed by the remaining
(non-contracted, non-batched) dimensions of `lhs`, then `rhs`.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_dot_general()`](https://r-xla.github.io/stablehlo/reference/hlo_dot_general.html).

## See also

[`nv_matmul()`](https://r-xla.github.io/anvil/reference/nv_matmul.md),
`%*%`

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
