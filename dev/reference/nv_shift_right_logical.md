# Logical Shift Right

Element-wise logical right bit shift.

## Usage

``` r
nv_shift_right_logical(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Left and right operand. Operands are [promoted to a common data
  type](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md).
  Scalars are
  [broadcast](https://r-xla.github.io/anvl/dev/reference/nv_broadcast_scalars.md)
  to the shape of the other operand.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and the promoted common data type of the inputs.

## See also

[`prim_shift_right_logical()`](https://r-xla.github.io/anvl/dev/reference/prim_shift_right_logical.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(8L, 16L, 32L))
y <- nv_array(c(1L, 2L, 3L))
nv_shift_right_logical(x, y)
#> AnvlArray
#>  4
#>  4
#>  4
#> [ CPUi32{3} ] 
```
