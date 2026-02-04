# Primitive Xor

Element-wise logical XOR. For a more user-friendly interface, see
[`nv_xor()`](https://r-xla.github.io/anvil/reference/nv_xor.md).

## Usage

``` r
nvl_xor(lhs, rhs)
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
[`stablehlo::hlo_xor()`](https://r-xla.github.io/stablehlo/reference/hlo_xor.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(TRUE, FALSE, TRUE))
  y <- nv_tensor(c(TRUE, TRUE, FALSE))
  nvl_xor(x, y)
})
#> AnvilTensor
#>  0
#>  1
#>  1
#> [ CPUi1{3} ] 
```
