# Primitive Tangent

Element-wise tangent.

## Usage

``` r
nvl_tan(operand)
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
[`stablehlo::hlo_tan()`](https://r-xla.github.io/stablehlo/reference/hlo_tan.html).

## See also

[`nv_tan()`](https://r-xla.github.io/anvil/reference/nv_tan.md),
[`tan()`](https://rdrr.io/r/base/Trig.html)

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(0, 0.5, 1))
  nvl_tan(x)
})
#> AnvilTensor
#>  0.0000
#>  0.5463
#>  1.5574
#> [ CPUf32{3} ] 
```
