# Primitive Exponential Minus One

Element-wise exp(x) - 1, more accurate for small x.

## Usage

``` r
nvl_expm1(operand)
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
[`stablehlo::hlo_exponential_minus_one()`](https://r-xla.github.io/stablehlo/reference/hlo_exponential_minus_one.html).

## See also

[`nv_expm1()`](https://r-xla.github.io/anvil/reference/nv_expm1.md)

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(0, 0.001, 1))
  nvl_expm1(x)
})
#> AnvilTensor
#>  0.0000
#>  0.0010
#>  1.7183
#> [ CPUf32{3} ] 
```
