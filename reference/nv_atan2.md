# Arctangent 2

Element-wise two-argument arctangent, i.e. the angle (in radians)
between the positive x-axis and the point `(rhs, lhs)`.

## Usage

``` r
nv_atan2(lhs, rhs)
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

[`nvl_atan2()`](https://r-xla.github.io/anvil/reference/nvl_atan2.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  y <- nv_tensor(c(1, 0, -1))
  x <- nv_tensor(c(0, 1, 0))
  nv_atan2(y, x)
})
#> AnvilTensor
#>   1.5708
#>   0.0000
#>  -1.5708
#> [ CPUf32{3} ] 
```
