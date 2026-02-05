# Transpose

Permutes the dimensions of a tensor. You can also use
[`t()`](https://rdrr.io/r/base/t.html) for matrices.

## Usage

``` r
nv_transpose(x, permutation = NULL)

# S3 method for class 'AnvilBox'
t(x)
```

## Arguments

- x:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensor to transpose.

- permutation:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  New ordering of dimensions. If `NULL` (default), reverses the
  dimensions.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same data type as `x` and shape `nv_shape(x)[permutation]`.

## See also

[`nvl_transpose()`](https://r-xla.github.io/anvil/reference/nvl_transpose.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(matrix(1:6, nrow = 2))
  t(x)
})
#> AnvilTensor
#>  1 2
#>  3 4
#>  5 6
#> [ CPUi32{3,2} ] 
```
