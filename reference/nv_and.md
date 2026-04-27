# Logical And

Element-wise logical AND. You can also use the `&` operator.

## Usage

``` r
nv_and(lhs, rhs)
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

[`prim_and()`](https://r-xla.github.io/anvl/reference/prim_and.md) for
the underlying primitive.

## Examples

``` r
x <- nv_array(c(TRUE, FALSE, TRUE))
y <- nv_array(c(TRUE, TRUE, FALSE))
x & y
#> AnvlArray
#>  1
#>  0
#>  0
#> [ CPUbool{3} ] 
```
