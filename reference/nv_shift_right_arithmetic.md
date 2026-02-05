# Arithmetic Shift Right

Element-wise arithmetic right bit shift.

## Usage

``` r
nv_shift_right_arithmetic(lhs, rhs)
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

[`nvl_shift_right_arithmetic()`](https://r-xla.github.io/anvil/reference/nvl_shift_right_arithmetic.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(8L, -16L, 32L))
  y <- nv_tensor(c(1L, 2L, 3L))
  nv_shift_right_arithmetic(x, y)
})
#> AnvilTensor
#>   4
#>  -4
#>   4
#> [ CPUi32{3} ] 
```
