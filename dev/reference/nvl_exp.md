# Primitive Exponential

Element-wise exponential.

## Usage

``` r
nvl_exp(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish value of data type floating-point.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`stablehlo::hlo_exponential()`](https://r-xla.github.io/stablehlo/reference/hlo_exponential.html).

## See also

[`nv_exp()`](https://r-xla.github.io/anvil/dev/reference/nv_exp.md),
[`exp()`](https://rdrr.io/r/base/Log.html)

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
jit_eval({
  x <- nv_array(c(0, 1, 2))
  nvl_exp(x)
})
}
```
