# Reverse

Reverses the order of elements along specified dimensions.

## Usage

``` r
nv_reverse(operand, dims)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions to reverse.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as `operand`.

## See also

[`nvl_reverse()`](https://r-xla.github.io/anvil/reference/nvl_reverse.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 2, 3, 4, 5))
  nv_reverse(x, dims = 1L)
})
#> AnvilTensor
#>  5
#>  4
#>  3
#>  2
#>  1
#> [ CPUf32{5} ] 
```
