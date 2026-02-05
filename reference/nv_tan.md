# Tangent

Element-wise tangent. You can also use
[`tan()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_tan(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_tan()`](https://r-xla.github.io/anvil/reference/nvl_tan.md) for
the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(0, 0.5, 1))
  tan(x)
})
#> AnvilTensor
#>  0.0000
#>  0.5463
#>  1.5574
#> [ CPUf32{3} ] 
```
