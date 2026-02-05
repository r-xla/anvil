# Primitive Cosine

Element-wise cosine.

## Usage

``` r
nvl_cosine(operand)
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
[`stablehlo::hlo_cosine()`](https://r-xla.github.io/stablehlo/reference/hlo_cosine.html).

## See also

[`nv_cosine()`](https://r-xla.github.io/anvil/reference/nv_cosine.md),
[`cos()`](https://rdrr.io/r/base/Trig.html)

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(0, pi / 2, pi))
  nvl_cosine(x)
})
#> AnvilTensor
#>   1.0000e+00
#>  -4.3711e-08
#>  -1.0000e+00
#> [ CPUf32{3} ] 
```
