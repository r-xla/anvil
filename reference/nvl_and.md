# Primitive And

Element-wise logical AND. For a more user-friendly interface, see
[`nv_and()`](https://r-xla.github.io/anvil/reference/nv_and.md), or use
the `&` operator.

## Usage

``` r
nvl_and(lhs, rhs)
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
[`stablehlo::hlo_and()`](https://r-xla.github.io/stablehlo/reference/hlo_and.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(TRUE, FALSE, TRUE))
  y <- nv_tensor(c(TRUE, TRUE, FALSE))
  nvl_and(x, y)
})
#> AnvilTensor
#>  1
#>  0
#>  0
#> [ CPUi1{3} ] 
```
