# Reciprocal Square Root

Element-wise reciprocal square root, i.e. `1 / sqrt(x)`.

## Usage

``` r
nv_rsqrt(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_rsqrt()`](https://r-xla.github.io/anvil/dev/reference/nvl_rsqrt.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(c(1, 4, 9))
  nv_rsqrt(x)
})
#> AnvilArray
#>  1.0000
#>  0.5000
#>  0.3333
#> [ CPUf32{3} ] 
```
