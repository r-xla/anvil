# Sign

Element-wise sign function. You can also use
[`sign()`](https://rdrr.io/r/base/sign.html).

## Usage

``` r
nv_sign(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_sign()`](https://r-xla.github.io/anvil/dev/reference/nvl_sign.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(c(-3, 0, 5))
  sign(x)
})
#> AnvilArray
#>  -1
#>   0
#>   1
#> [ CPUf32{3} ] 
```
