# Cholesky Decomposition

Computes the Cholesky decomposition of a symmetric positive-definite
matrix. Supports batched inputs: dimensions before the last two are
batch dimensions.

## Usage

``` r
nv_chol(operand, lower = FALSE)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Symmetric positive-definite matrix with at least 2 dimensions. The
  last two dimensions form the square matrix; any leading dimensions are
  batch dimensions.

- lower:

  (`logical(1)`)  
  If `FALSE` (default, matching base R's
  [`base::chol()`](https://rdrr.io/r/base/chol.html)), compute the upper
  triangular factor `U` such that `operand = t(U) %*% U`. If `TRUE`,
  compute the lower triangular factor `L` such that
  `operand = L %*% t(L)`.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Triangular matrix with the same shape and data type as the input.

## See also

[`nv_solve()`](https://r-xla.github.io/anvl/dev/reference/nv_solve.md),
[`prim_chol()`](https://r-xla.github.io/anvl/dev/reference/prim_chol.md)

## Examples

``` r
a <- nv_matrix(c(4, 2, 2, 3), nrow = 2, dtype = "f32")
nv_chol(a)
#> AnvlArray
#>  2.0000 1.0000
#>  0.0000 1.4142
#> [ CPUf32{2,2} ] 
```
