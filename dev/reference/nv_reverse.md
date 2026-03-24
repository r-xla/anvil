# Reverse

Reverses the order of elements along specified dimensions.

## Usage

``` r
nv_reverse(operand, dims)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions to reverse.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as `operand`.

## See also

[`nvl_reverse()`](https://r-xla.github.io/anvil/dev/reference/nvl_reverse.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(c(1, 2, 3, 4, 5))
  nv_reverse(x, dims = 1L)
})
#> AnvilArray
#>  5
#>  4
#>  3
#>  2
#>  1
#> [ CPUf32{5} ] 
```
