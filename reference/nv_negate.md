# Negation

Negates a tensor element-wise. You can also use the unary `-` operator.

## Usage

``` r
nv_negate(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_negate()`](https://r-xla.github.io/anvil/reference/nvl_negate.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, -2, 3))
  -x
})
#> AnvilTensor
#>  -1
#>   2
#>  -3
#> [ CPUf32{3} ] 
```
