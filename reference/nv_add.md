# Addition

Adds two tensors element-wise. You can also use the `+` operator.

## Usage

``` r
nv_add(lhs, rhs)
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

[`nvl_add()`](https://r-xla.github.io/anvil/reference/nvl_add.md) for
the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 2, 3))
  y <- nv_tensor(c(4, 5, 6))
  x + y
})
#> AnvilTensor
#>  5
#>  7
#>  9
#> [ CPUf32{3} ] 
```
