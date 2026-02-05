# Power

Raises `lhs` to the power of `rhs` element-wise. You can also use the
`^` operator.

## Usage

``` r
nv_pow(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Left and right operand. Operands are [promoted to a common data
  type](https://r-xla.github.io/anvil/reference/nv_promote_to_common.md).
  Scalars are
  [broadcast](https://r-xla.github.io/anvil/reference/nv_broadcast_scalars.md)
  to the shape of the other operand.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and the promoted common data type of the inputs.

## See also

[`nvl_pow()`](https://r-xla.github.io/anvil/reference/nvl_pow.md) for
the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(2, 3, 4))
  y <- nv_tensor(c(3, 2, 1))
  x ^ y
})
#> AnvilTensor
#>  8
#>  9
#>  4
#> [ CPUf32{3} ] 
```
