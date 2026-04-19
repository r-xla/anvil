# Tangent

Element-wise tangent. You can also use
[`tan()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_tan(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_tan()`](https://r-xla.github.io/anvil/dev/reference/nvl_tan.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(c(0, 0.5, 1))
  tan(x)
})
#> AnvilArray
#>  0.0000
#>  0.5463
#>  1.5574
#> [ CPUf32{3} ] 
```
