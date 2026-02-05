# Primitive Or

Element-wise logical OR.

## Usage

``` r
nvl_or(lhs, rhs)
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
[`stablehlo::hlo_or()`](https://r-xla.github.io/stablehlo/reference/hlo_or.html).

## See also

[`nv_or()`](https://r-xla.github.io/anvil/reference/nv_or.md), `|`

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(TRUE, FALSE, TRUE))
  y <- nv_tensor(c(TRUE, TRUE, FALSE))
  nvl_or(x, y)
})
#> AnvilTensor
#>  1
#>  1
#>  1
#> [ CPUi1{3} ] 
```
