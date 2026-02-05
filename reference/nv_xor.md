# Logical Xor

Element-wise logical XOR.

## Usage

``` r
nv_xor(lhs, rhs)
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

[`nvl_xor()`](https://r-xla.github.io/anvil/reference/nvl_xor.md) for
the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(TRUE, FALSE, TRUE))
  y <- nv_tensor(c(TRUE, TRUE, FALSE))
  nv_xor(x, y)
})
#> AnvilTensor
#>  0
#>  1
#>  1
#> [ CPUi1{3} ] 
```
