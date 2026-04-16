# Primitive Logistic (Sigmoid)

Element-wise logistic sigmoid: 1 / (1 + exp(-x)).

## Usage

``` r
nvl_logistic(operand)
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
[`stablehlo::hlo_logistic()`](https://r-xla.github.io/stablehlo/reference/hlo_logistic.html).

## See also

[`nv_logistic()`](https://r-xla.github.io/anvil/dev/reference/nv_logistic.md)

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
jit_eval({
  x <- nv_array(c(-2, 0, 2))
  nvl_logistic(x)
})
}
```
