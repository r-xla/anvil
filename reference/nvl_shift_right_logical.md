# Primitive Logical Shift Right

Element-wise logical right bit shift. For a more user-friendly
interface, see
[`nv_shift_right_logical()`](https://r-xla.github.io/anvil/reference/nv_shift_right_logical.md).

## Usage

``` r
nvl_shift_right_logical(lhs, rhs)
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
[`stablehlo::hlo_shift_right_logical()`](https://r-xla.github.io/stablehlo/reference/hlo_shift_right_logical.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(8L, 16L, 32L))
  y <- nv_tensor(c(1L, 2L, 3L))
  nvl_shift_right_logical(x, y)
})
#> AnvilTensor
#>  4
#>  4
#>  4
#> [ CPUi32{3} ] 
```
