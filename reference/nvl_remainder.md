# Primitive Remainder

Element-wise remainder of division.

## Usage

``` r
nvl_remainder(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish values of data type integer, unsigned integer, or
  floating-point. Must have the same shape.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the inputs. It is ambiguous if both
inputs are ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_remainder()`](https://r-xla.github.io/stablehlo/reference/hlo_remainder.html).

## See also

[`nv_remainder()`](https://r-xla.github.io/anvil/reference/nv_remainder.md),
`%%`

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(7, 10, 15))
  y <- nv_tensor(c(3, 4, 6))
  nvl_remainder(x, y)
})
#> AnvilTensor
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
