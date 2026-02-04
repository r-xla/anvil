# Primitive Shift Left

Element-wise left bit shift. For a more user-friendly interface, see
[`nv_shift_left()`](https://r-xla.github.io/anvil/reference/nv_shift_left.md).

## Usage

``` r
nvl_shift_left(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish values of data type boolean, integer, or unsigned integer.
  Must have the same shape.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the inputs. It is ambiguous if both
inputs are ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_shift_left()`](https://r-xla.github.io/stablehlo/reference/hlo_shift_left.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1L, 2L, 4L))
  y <- nv_tensor(c(1L, 2L, 1L))
  nvl_shift_left(x, y)
})
#> AnvilTensor
#>  2
#>  8
#>  8
#> [ CPUi32{3} ] 
```
