# Equal

Element-wise equality comparison. You can also use the `==` operator.

## Usage

``` r
nv_eq(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`tensorish`](https://r-xla.github.io/anvil/dev/reference/tensorish.md))  
  Left and right operand. Operands are [promoted to a common data
  type](https://r-xla.github.io/anvil/dev/reference/nv_promote_to_common.md).
  Scalars are
  [broadcast](https://r-xla.github.io/anvil/dev/reference/nv_broadcast_scalars.md)
  to the shape of the other operand.

## Value

[`tensorish`](https://r-xla.github.io/anvil/dev/reference/tensorish.md)  
Has the same shape as the inputs and boolean data type.

## See also

[`nvl_eq()`](https://r-xla.github.io/anvil/dev/reference/nvl_eq.md) for
the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 2, 3))
  y <- nv_tensor(c(1, 3, 2))
  x == y
})
#> AnvilTensor
#>  1
#>  0
#>  0
#> [ CPUbool{3} ] 
```
