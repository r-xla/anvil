# Mean Reduction

Computes the arithmetic mean along the specified dimensions.

## Usage

``` r
nv_reduce_mean(operand, dims, drop = TRUE)
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

## Details

Implemented as `nv_reduce_sum(operand, dims, drop) / n` where `n` is the
product of the reduced dimension sizes.

## See also

[`nv_reduce_sum()`](https://r-xla.github.io/anvil/reference/nv_reduce_sum.md)

## Examples

``` r
jit_eval({
  x <- nv_tensor(matrix(1:6, nrow = 2))
  nv_reduce_mean(x, dims = 1L)
})
#> AnvilTensor
#>  1.5000
#>  3.5000
#>  5.5000
#> [ CPUf32?{3} ] 
```
