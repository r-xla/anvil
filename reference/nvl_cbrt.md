# Primitive Cube Root

Element-wise cube root.

## Usage

``` r
nvl_cbrt(operand)
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
[`stablehlo::hlo_cbrt()`](https://r-xla.github.io/stablehlo/reference/hlo_cbrt.html).

## See also

[`nv_cbrt()`](https://r-xla.github.io/anvil/reference/nv_cbrt.md)

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 8, 27))
  nvl_cbrt(x)
})
#> AnvilTensor
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
