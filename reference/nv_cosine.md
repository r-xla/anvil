# Cosine

Element-wise cosine. You can also use
[`cos()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_cosine(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_cosine()`](https://r-xla.github.io/anvil/reference/nvl_cosine.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(0, pi / 2, pi))
  cos(x)
})
#> AnvilTensor
#>   1.0000e+00
#>  -4.3711e-08
#>  -1.0000e+00
#> [ CPUf32{3} ] 
```
