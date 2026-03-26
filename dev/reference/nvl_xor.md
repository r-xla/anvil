# Primitive Xor

Element-wise logical XOR.

## Usage

``` r
nvl_xor(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish values of data type boolean, integer, or unsigned integer.
  Must have the same shape.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the inputs. It is ambiguous if both
inputs are ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`stablehlo::hlo_xor()`](https://r-xla.github.io/stablehlo/reference/hlo_xor.html).

## See also

[`nv_xor()`](https://r-xla.github.io/anvil/dev/reference/nv_xor.md)

## Examples

``` r
jit_eval({
  x <- nv_array(c(TRUE, FALSE, TRUE))
  y <- nv_array(c(TRUE, TRUE, FALSE))
  nvl_xor(x, y)
})
#> AnvilArray
#>  0
#>  1
#>  1
#> [ CPUbool{3} ] 
```
