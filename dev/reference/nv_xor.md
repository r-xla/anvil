# Logical Xor

Element-wise logical XOR.

## Usage

``` r
nv_xor(lhs, rhs)
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

[`nvl_xor()`](https://r-xla.github.io/anvil/dev/reference/nvl_xor.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(c(TRUE, FALSE, TRUE))
  y <- nv_array(c(TRUE, TRUE, FALSE))
  nv_xor(x, y)
})
#> AnvilArray
#>  0
#>  1
#>  1
#> [ CPUbool{3} ] 
```
