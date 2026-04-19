# Transpose

Permutes the dimensions of an array. You can also use
[`t()`](https://rdrr.io/r/base/t.html) for matrices.

## Usage

``` r
nv_transpose(x, permutation = NULL)

# S3 method for class 'AnvilBox'
t(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Array to transpose.

- permutation:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  New ordering of dimensions. If `NULL` (default), reverses the
  dimensions.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same data type as `x` and shape `nv_shape(x)[permutation]`.

## See also

[`nvl_transpose()`](https://r-xla.github.io/anvil/dev/reference/nvl_transpose.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(matrix(1:6, nrow = 2))
  t(x)
})
#> AnvilArray
#>  1 2
#>  3 4
#>  5 6
#> [ CPUi32{3,2} ] 
```
