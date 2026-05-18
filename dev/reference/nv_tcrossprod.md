# Transpose Cross Product (Matrix)

Computes `x %*% t(y)`. If `y` is missing, computes `x %*% t(x)`.

## Usage

``` r
nv_tcrossprod(x, y = NULL)

# S3 method for class 'AnvlBox'
tcrossprod(x, y = NULL, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  An array with at least 2 dimensions.

- y:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
  \| `NULL`)  
  Optional second array. If `NULL`, uses `x`.

- ...:

  Unused.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)

## See also

[`nv_crossprod()`](https://r-xla.github.io/anvl/dev/reference/nv_crossprod.md),
[`nv_matmul()`](https://r-xla.github.io/anvl/dev/reference/nv_matmul.md)

## Examples

``` r
x <- nv_matrix(1:6, nrow = 2, dtype = "f32")
nv_tcrossprod(x)
#> AnvlArray
#>  35 44
#>  44 56
#> [ CPUf32{2,2} ] 
```
