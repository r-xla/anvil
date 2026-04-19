# Shift Left

Element-wise left bit shift.

## Usage

``` r
nv_shift_left(lhs, rhs)
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

[`nvl_shift_left()`](https://r-xla.github.io/anvil/dev/reference/nvl_shift_left.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(c(1L, 2L, 4L))
  y <- nv_array(c(1L, 2L, 1L))
  nv_shift_left(x, y)
})
#> AnvilArray
#>  2
#>  8
#>  8
#> [ CPUi32{3} ] 
```
