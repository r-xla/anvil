# Primitive Atan2

Element-wise atan2 operation. For a more user-friendly interface, see
[`nv_atan2()`](https://r-xla.github.io/anvil/reference/nv_atan2.md).

## Usage

``` r
nvl_atan2(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish values of data type floating-point. Must have the same
  shape.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the inputs. It is ambiguous if both
inputs are ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_atan2()`](https://r-xla.github.io/stablehlo/reference/hlo_atan2.html).

## Examples

``` r
jit_eval({
  y <- nv_tensor(c(1, 0, -1))
  x <- nv_tensor(c(0, 1, 0))
  nvl_atan2(y, x)
})
#> AnvilTensor
#>   1.5708
#>   0.0000
#>  -1.5708
#> [ CPUf32{3} ] 
```
