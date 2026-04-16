# Primitive Sine

Element-wise sine.

## Usage

``` r
nvl_sine(operand)
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
[`stablehlo::hlo_sine()`](https://r-xla.github.io/stablehlo/reference/hlo_sine.html).

## See also

[`nv_sine()`](https://r-xla.github.io/anvil/dev/reference/nv_sine.md),
[`sin()`](https://rdrr.io/r/base/Trig.html)

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
jit_eval({
  x <- nv_array(c(0, pi / 2, pi))
  nvl_sine(x)
})
}
```
