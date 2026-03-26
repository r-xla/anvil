# Primitive Arithmetic Shift Right

Element-wise arithmetic right bit shift.

## Usage

``` r
nvl_shift_right_arithmetic(lhs, rhs)
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

- `reverse`

## StableHLO

Lowers to
[`stablehlo::hlo_shift_right_arithmetic()`](https://r-xla.github.io/stablehlo/reference/hlo_shift_right_arithmetic.html).

## See also

[`nv_shift_right_arithmetic()`](https://r-xla.github.io/anvil/dev/reference/nv_shift_right_arithmetic.md)

## Examples

``` r
jit_eval({
  x <- nv_array(c(8L, -16L, 32L))
  y <- nv_array(c(1L, 2L, 3L))
  nvl_shift_right_arithmetic(x, y)
})
#> AnvilArray
#>   4
#>  -4
#>   4
#> [ CPUi32{3} ] 
```
