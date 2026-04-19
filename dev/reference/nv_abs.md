# Absolute Value

Element-wise absolute value. You can also use
[`abs()`](https://rdrr.io/r/base/MathFun.html).

## Usage

``` r
nv_abs(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_abs()`](https://r-xla.github.io/anvil/dev/reference/nvl_abs.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(c(-1, 2, -3))
  abs(x)
})
#> AnvilArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
