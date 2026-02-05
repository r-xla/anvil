# Primitive Ifelse

Element-wise selection based on a boolean predicate, like R's
[`ifelse()`](https://rdrr.io/r/base/ifelse.html). For each element,
returns the corresponding element from `true_value` where `pred` is
`TRUE` and from `false_value` where `pred` is `FALSE`.

## Usage

``` r
nvl_ifelse(pred, true_value, false_value)
```

## Arguments

- pred:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
  of boolean type)  
  Predicate tensor. Must be scalar or have the same shape as
  `true_value`.

- true_value, false_value:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Values to select from. Must have the same dtype and shape.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same dtype and shape as `true_value`. It is ambiguous if both
`true_value` and `false_value` are ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_select()`](https://r-xla.github.io/stablehlo/reference/hlo_select.html).

## See also

[`nv_ifelse()`](https://r-xla.github.io/anvil/reference/nv_ifelse.md)

## Examples

``` r
jit_eval({
  pred <- nv_tensor(c(TRUE, FALSE, TRUE))
  nvl_ifelse(pred, nv_tensor(c(1, 2, 3)), nv_tensor(c(4, 5, 6)))
})
#> AnvilTensor
#>  1
#>  5
#>  3
#> [ CPUf32{3} ] 
```
