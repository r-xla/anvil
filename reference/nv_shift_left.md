# Shift Left

Element-wise left bit shift.

## Usage

``` r
nv_shift_left(lhs, rhs)
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

[`nvl_shift_left()`](https://r-xla.github.io/anvil/reference/nvl_shift_left.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1L, 2L, 4L))
  y <- nv_tensor(c(1L, 2L, 1L))
  nv_shift_left(x, y)
})
#> AnvilTensor
#>  2
#>  8
#>  8
#> [ CPUi32{3} ] 
```
