# Natural Logarithm

Element-wise natural logarithm. You can also use
[`log()`](https://rdrr.io/r/base/Log.html).

## Usage

``` r
nv_log(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_log()`](https://r-xla.github.io/anvil/dev/reference/nvl_log.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(c(1, 2.718, 7.389))
  log(x)
})
#> AnvilArray
#>  0.0000
#>  0.9999
#>  2.0000
#> [ CPUf32{3} ] 
```
