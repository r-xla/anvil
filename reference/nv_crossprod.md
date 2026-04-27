# Cross Product (Matrix)

Computes `t(x) %*% y`. If `y` is missing, computes `t(x) %*% x`.

## Usage

``` r
nv_crossprod(x, y = NULL)

# S3 method for class 'AnvlBox'
crossprod(x, y = NULL, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  An array with at least 2 dimensions.

- y:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md) \|
  `NULL`)  
  Optional second array. If `NULL`, uses `x`.

- ...:

  Unused.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)

## See also

[`nv_tcrossprod()`](https://r-xla.github.io/anvl/reference/nv_tcrossprod.md),
[`nv_matmul()`](https://r-xla.github.io/anvl/reference/nv_matmul.md)

## Examples

``` r
x <- nv_array(matrix(1:6, nrow = 3), dtype = "f32")
nv_crossprod(x)
#> AnvlArray
#>  14 32
#>  32 77
#> [ CPUf32{2,2} ] 
```
