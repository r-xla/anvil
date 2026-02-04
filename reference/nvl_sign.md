# Primitive Sign

Element-wise sign. Is the same as
[`nv_sign()`](https://r-xla.github.io/anvil/reference/nv_sign.md). You
can also use [`sign()`](https://rdrr.io/r/base/sign.html).

## Usage

``` r
nvl_sign(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of data type signed integer or floating-point.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_sign()`](https://r-xla.github.io/stablehlo/reference/hlo_sign.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(-3, 0, 5))
  nvl_sign(x)
})
#> AnvilTensor
#>  -1
#>   0
#>   1
#> [ CPUf32{3} ] 
```
