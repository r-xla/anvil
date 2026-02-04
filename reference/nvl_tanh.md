# Primitive Hyperbolic Tangent

Element-wise hyperbolic tangent. Is the same as
[`nv_tanh()`](https://r-xla.github.io/anvil/reference/nv_tanh.md). You
can also use [`tanh()`](https://rdrr.io/r/base/Hyperbolic.html).

## Usage

``` r
nvl_tanh(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of data type floating-point.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_tanh()`](https://r-xla.github.io/stablehlo/reference/hlo_tanh.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(-1, 0, 1))
  nvl_tanh(x)
})
#> AnvilTensor
#>  -0.7616
#>   0.0000
#>   0.7616
#> [ CPUf32{3} ] 
```
