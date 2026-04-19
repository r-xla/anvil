# Negation

Negates an array element-wise. You can also use the unary `-` operator.

## Usage

``` r
nv_negate(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_negate()`](https://r-xla.github.io/anvil/dev/reference/nvl_negate.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(c(1, -2, 3))
  -x
})
#> AnvilArray
#>  -1
#>   2
#>  -3
#> [ CPUf32{3} ] 
```
