# Square Root

Element-wise square root. You can also use
[`sqrt()`](https://rdrr.io/r/base/MathFun.html).

## Usage

``` r
nv_sqrt(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_sqrt()`](https://r-xla.github.io/anvil/dev/reference/nvl_sqrt.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(c(1, 4, 9))
  sqrt(x)
})
#> AnvilArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
