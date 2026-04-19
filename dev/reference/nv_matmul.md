# Matrix Multiplication

Matrix multiplication of two arrays. You can also use the `%*%`
operator. Supports batched matrix multiplication when inputs have more
than 2 dimensions.

## Usage

``` r
nv_matmul(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrays with at least 2 dimensions. Operands are [promoted to a common
  data
  type](https://r-xla.github.io/anvil/dev/reference/nv_promote_to_common.md).

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)

## Shapes

- `lhs`: `(b1, ..., bk, m, n)`

- `rhs`: `(b1, ..., bk, n, p)`

- output: `(b1, ..., bk, m, p)`

## See also

[`nvl_dot_general()`](https://r-xla.github.io/anvil/dev/reference/nvl_dot_general.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(matrix(1:6, nrow = 2))
  y <- nv_array(matrix(1:6, nrow = 3))
  x %*% y
})
#> AnvilArray
#>  22 49
#>  28 64
#> [ CPUi32{2,2} ] 
```
