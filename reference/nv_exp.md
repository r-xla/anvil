# Exponential

Element-wise exponential. You can also use
[`exp()`](https://rdrr.io/r/base/Log.html).

## Usage

``` r
nv_exp(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_exp()`](https://r-xla.github.io/anvil/reference/nvl_exp.md) for
the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(0, 1, 2))
  exp(x)
})
#> AnvilTensor
#>  1.0000
#>  2.7183
#>  7.3891
#> [ CPUf32{3} ] 
```
