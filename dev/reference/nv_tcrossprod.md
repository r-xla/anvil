# Transpose Cross Product (Matrix)

Computes `x %*% t(y)`. If `y` is missing, computes `x %*% t(x)`.

## Usage

``` r
nv_tcrossprod(x, y = NULL)

# S3 method for class 'AnvilBox'
tcrossprod(x, y = NULL, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  An array with at least 2 dimensions.

- y:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)
  \| `NULL`)  
  Optional second array. If `NULL`, uses `x`.

- ...:

  Unused.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)

## See also

[`nv_crossprod()`](https://r-xla.github.io/anvil/dev/reference/nv_crossprod.md),
[`nv_matmul()`](https://r-xla.github.io/anvil/dev/reference/nv_matmul.md)

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
jit_eval({
  x <- nv_array(matrix(1:6, nrow = 2), dtype = "f32")
  nv_tcrossprod(x)
})
}
```
