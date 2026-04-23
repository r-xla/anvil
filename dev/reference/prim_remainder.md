# Primitive Remainder

Element-wise remainder of division.

## Usage

``` r
prim_remainder(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrayish values of data type integer, unsigned integer, or
  floating-point. Must have the same shape.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same shape and data type as the inputs. It is ambiguous if both
inputs are ambiguous.

## Implemented Rules

- `stablehlo`

- `reverse`

## StableHLO

Lowers to
[`stablehlo::hlo_remainder()`](https://r-xla.github.io/stablehlo/reference/hlo_remainder.html).

## See also

[`nv_remainder()`](https://r-xla.github.io/anvl/dev/reference/nv_remainder.md),
`%%`

## Examples

``` r
x <- nv_array(c(7, 10, 15))
y <- nv_array(c(3, 4, 6))
prim_remainder(x, y)
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
