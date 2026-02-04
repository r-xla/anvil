# Primitive Sine

Element-wise sine. Is the same as
[`nv_sine()`](https://r-xla.github.io/anvil/reference/nv_sine.md). You
can also use [`sin()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nvl_sine(operand)
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
[`stablehlo::hlo_sine()`](https://r-xla.github.io/stablehlo/reference/hlo_sine.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(0, pi / 2, pi))
  nvl_sine(x)
})
#> AnvilTensor
#>   0.0000e+00
#>   1.0000e+00
#>  -8.7423e-08
#> [ CPUf32{3} ] 
```
