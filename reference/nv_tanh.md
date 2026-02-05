# Hyperbolic Tangent

Element-wise hyperbolic tangent. You can also use
[`tanh()`](https://rdrr.io/r/base/Hyperbolic.html).

## Usage

``` r
nv_tanh(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_tanh()`](https://r-xla.github.io/anvil/reference/nvl_tanh.md) for
the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(-1, 0, 1))
  tanh(x)
})
#> AnvilTensor
#>  -0.7616
#>   0.0000
#>   0.7616
#> [ CPUf32{3} ] 
```
