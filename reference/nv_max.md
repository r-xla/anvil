# Maximum

Element-wise maximum of two tensors.

## Usage

``` r
nv_max(lhs, rhs)
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

[`nvl_max()`](https://r-xla.github.io/anvil/reference/nvl_max.md) for
the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 5, 3))
  y <- nv_tensor(c(4, 2, 6))
  nv_max(x, y)
})
#> AnvilTensor
#>  4
#>  5
#>  6
#> [ CPUf32{3} ] 
```
