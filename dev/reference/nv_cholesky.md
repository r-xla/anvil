# Cholesky Decomposition

Computes the Cholesky decomposition of a symmetric positive-definite
matrix. Supports batched inputs: dimensions before the last two are
batch dimensions.

## Usage

``` r
nv_cholesky(a, lower = TRUE)
```

## Arguments

- a:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Symmetric positive-definite matrix with at least 2 dimensions. The
  last two dimensions form the square matrix; any leading dimensions are
  batch dimensions.

- lower:

  (`logical(1)`)  
  If `TRUE` (default), compute the lower triangular factor `L` such that
  `a = L %*% t(L)`. If `FALSE`, compute the upper triangular factor `U`
  such that `a = t(U) %*% U`.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Triangular matrix with the same shape and data type as the input.

## See also

[`nv_solve()`](https://r-xla.github.io/anvil/dev/reference/nv_solve.md),
[`nvl_cholesky()`](https://r-xla.github.io/anvil/dev/reference/nvl_cholesky.md)

## Examples

``` r
jit_eval({
  a <- nv_array(matrix(c(4, 2, 2, 3), nrow = 2), dtype = "f32")
  nv_cholesky(a)
})
#> AnvilArray
#>  2.0000 0.0000
#>  1.0000 1.4142
#> [ CPUf32{2,2} ] 
```
