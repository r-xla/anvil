# Cross Product (Matrix)

Computes `t(x) %*% y`. If `y` is missing, computes `t(x) %*% x`.

## Usage

``` r
nv_crossprod(x, y = NULL)

# S3 method for class 'AnvilBox'
crossprod(x, y = NULL, ...)
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

[`nv_tcrossprod()`](https://r-xla.github.io/anvil/dev/reference/nv_tcrossprod.md),
[`nv_matmul()`](https://r-xla.github.io/anvil/dev/reference/nv_matmul.md)

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
jit_eval({
  x <- nv_array(matrix(1:6, nrow = 3), dtype = "f32")
  nv_crossprod(x)
})
}
```
