# Logical Xor

Element-wise logical XOR.

## Usage

``` r
nv_xor(lhs, rhs)
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

[`prim_xor()`](https://r-xla.github.io/anvl/reference/prim_xor.md) for
the underlying primitive.

## Examples

``` r
x <- nv_array(c(TRUE, FALSE, TRUE))
y <- nv_array(c(TRUE, TRUE, FALSE))
nv_xor(x, y)
#> AnvlArray
#>  0
#>  1
#>  1
#> [ CPUbool{3} ] 
```
