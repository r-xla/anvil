# Remainder

Element-wise remainder of division. You can also use the `%%` operator.

## Usage

``` r
nv_remainder(lhs, rhs)
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

[`nvl_remainder()`](https://r-xla.github.io/anvil/reference/nvl_remainder.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(7, 8, 9))
  y <- nv_tensor(c(3, 3, 4))
  x %% y
})
#> AnvilTensor
#>  1
#>  2
#>  1
#> [ CPUf32{3} ] 
```
