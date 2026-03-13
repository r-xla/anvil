# Solve Linear System

Solves the linear system `a %*% x = b` for `x`, where `a` is a symmetric
positive-definite matrix. Uses Cholesky decomposition internally.
Supports batched inputs: `a` and `b` must have the same batch dimensions
(all dimensions before the last two).

## Usage

``` r
nv_solve(a, b)
```

## Arguments

- a:

  ([`tensorish`](https://r-xla.github.io/anvil/dev/reference/tensorish.md))  
  Symmetric positive-definite matrix.

- b:

  ([`tensorish`](https://r-xla.github.io/anvil/dev/reference/tensorish.md))  
  Right-hand side matrix or vector. Must have the same data type and
  batch dimensions as `a`.

## Value

[`tensorish`](https://r-xla.github.io/anvil/dev/reference/tensorish.md)  
The solution `x` such that `a %*% x = b`.

## Shapes

- `a`: `(..., n, n)`

- `b`: `(..., n, k)`

- output: same shape as `b`

where `...` are zero or more batch dimensions that must match between
`a` and `b`.

## See also

[`nv_cholesky()`](https://r-xla.github.io/anvil/dev/reference/nv_cholesky.md),
[`nvl_cholesky()`](https://r-xla.github.io/anvil/dev/reference/nvl_cholesky.md),
[`nvl_triangular_solve()`](https://r-xla.github.io/anvil/dev/reference/nvl_triangular_solve.md)

## Examples

``` r
jit_eval({
  a <- nv_tensor(matrix(c(4, 2, 2, 3), nrow = 2), dtype = "f32")
  b <- nv_tensor(matrix(c(1, 2), nrow = 2), dtype = "f32")
  nv_solve(a, b)
})
#> AnvilTensor
#>  -0.1250
#>   0.7500
#> [ CPUf32{2,1} ] 
```
