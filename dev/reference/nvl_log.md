# Primitive Logarithm

Element-wise natural logarithm.

## Usage

``` r
nvl_log(operand)
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
[`stablehlo::hlo_log()`](https://r-xla.github.io/stablehlo/reference/hlo_log.html).

## See also

[`nv_log()`](https://r-xla.github.io/anvil/dev/reference/nv_log.md),
[`log()`](https://rdrr.io/r/base/Log.html)

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
jit_eval({
  x <- nv_array(c(1, 2.718, 7.389))
  nvl_log(x)
})
}
```
