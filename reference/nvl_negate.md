# Primitive Negation

Negates a tensor element-wise.

## Usage

``` r
nvl_negate(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of data type integer or floating-point.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_negate()`](https://r-xla.github.io/stablehlo/reference/hlo_negate.html).

## See also

[`nv_negate()`](https://r-xla.github.io/anvil/reference/nv_negate.md),
unary `-`

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, -2, 3))
  nvl_negate(x)
})
#> AnvilTensor
#>  -1
#>   2
#>  -3
#> [ CPUf32{3} ] 
```
