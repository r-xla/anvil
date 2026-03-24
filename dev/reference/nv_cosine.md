# Cosine

Element-wise cosine. You can also use
[`cos()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_cosine(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_cosine()`](https://r-xla.github.io/anvil/dev/reference/nvl_cosine.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(c(0, pi / 2, pi))
  cos(x)
})
#> AnvilArray
#>   1.0000e+00
#>  -4.3711e-08
#>  -1.0000e+00
#> [ CPUf32{3} ] 
```
