# Minimum

Element-wise minimum of two arrays.

## Usage

``` r
nv_min(lhs, rhs)
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

[`prim_min()`](https://r-xla.github.io/anvl/reference/prim_min.md) for
the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 5, 3))
y <- nv_array(c(4, 2, 6))
nv_min(x, y)
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
