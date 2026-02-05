# Sine

Element-wise sine. You can also use
[`sin()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_sine(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_sine()`](https://r-xla.github.io/anvil/reference/nvl_sine.md) for
the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(0, pi / 2, pi))
  sin(x)
})
#> AnvilTensor
#>   0.0000e+00
#>   1.0000e+00
#>  -8.7423e-08
#> [ CPUf32{3} ] 
```
