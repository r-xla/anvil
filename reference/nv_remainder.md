# Remainder

Element-wise remainder of division. You can also use the `%%` operator.

## Usage

``` r
nv_remainder(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Left and right operand. Operands are [promoted to a common data
  type](https://r-xla.github.io/anvl/reference/nv_promote_to_common.md).
  Scalars are
  [broadcast](https://r-xla.github.io/anvl/reference/nv_broadcast_scalars.md)
  to the shape of the other operand.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and the promoted common data type of the inputs.

## See also

[`prim_remainder()`](https://r-xla.github.io/anvl/reference/prim_remainder.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(c(7, 8, 9))
y <- nv_array(c(3, 3, 4))
x %% y
#> AnvlArray
#>  1
#>  2
#>  1
#> [ CPUf32{3} ] 
```
