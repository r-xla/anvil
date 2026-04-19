# Sine

Element-wise sine. You can also use
[`sin()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_sine(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_sine()`](https://r-xla.github.io/anvil/dev/reference/nvl_sine.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(c(0, pi / 2, pi))
  sin(x)
})
#> AnvilArray
#>   0.0000e+00
#>   1.0000e+00
#>  -8.7423e-08
#> [ CPUf32{3} ] 
```
