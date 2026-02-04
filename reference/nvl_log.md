# Primitive Logarithm

Element-wise natural logarithm. Is the same as
[`nv_log()`](https://r-xla.github.io/anvil/reference/nv_log.md). You can
also use [`log()`](https://rdrr.io/r/base/Log.html).

## Usage

``` r
nvl_log(operand)
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
[`stablehlo::hlo_log()`](https://r-xla.github.io/stablehlo/reference/hlo_log.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 2.718, 7.389))
  nvl_log(x)
})
#> AnvilTensor
#>  0.0000
#>  0.9999
#>  2.0000
#> [ CPUf32{3} ] 
```
