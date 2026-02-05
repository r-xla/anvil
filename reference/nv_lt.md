# Less Than

Element-wise less than comparison. You can also use the `<` operator.

## Usage

``` r
nv_lt(lhs, rhs)
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
Has the same shape as the inputs and boolean data type.

## See also

[`nvl_lt()`](https://r-xla.github.io/anvil/reference/nvl_lt.md) for the
underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 2, 3))
  y <- nv_tensor(c(3, 2, 1))
  x < y
})
#> AnvilTensor
#>  1
#>  0
#>  0
#> [ CPUi1{3} ] 
```
