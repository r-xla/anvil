# Max Reduction

Finds the maximum of tensor elements along the specified dimensions.

## Usage

``` r
nv_reduce_max(operand, dims, drop = TRUE)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions to reduce.

- drop:

  (`logical(1)`)  
  Whether to drop reduced dimensions.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same data type as the input. When `drop = TRUE`, the reduced
dimensions are removed. When `drop = FALSE`, the reduced dimensions are
set to 1.

## See also

[`nvl_reduce_max()`](https://r-xla.github.io/anvil/reference/nvl_reduce_max.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(matrix(1:6, nrow = 2))
  nv_reduce_max(x, dims = 1L)
})
#> AnvilTensor
#>  2
#>  4
#>  6
#> [ CPUi32{3} ] 
```
