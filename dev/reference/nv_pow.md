# Power

Raises `lhs` to the power of `rhs` element-wise. You can also use the
`^` operator.

## Usage

``` r
nv_pow(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Left and right operand. Operands are [promoted to a common data
  type](https://r-xla.github.io/anvil/dev/reference/nv_promote_to_common.md).
  Scalars are
  [broadcast](https://r-xla.github.io/anvil/dev/reference/nv_broadcast_scalars.md)
  to the shape of the other operand.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and the promoted common data type of the inputs.

## See also

[`nvl_pow()`](https://r-xla.github.io/anvil/dev/reference/nvl_pow.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(c(2, 3, 4))
  y <- nv_array(c(3, 2, 1))
  x ^ y
})
#> AnvilArray
#>  8
#>  9
#>  4
#> [ CPUf32{3} ] 
```
