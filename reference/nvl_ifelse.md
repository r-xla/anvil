# Primitive Ifelse

Selects elements based on a predicate.

## Usage

``` r
nvl_ifelse(pred, true_value, false_value)
```

## Arguments

- pred:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
  of boolean type)  
  Predicate tensor.

- true_value:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Value when pred is true.

- false_value:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Value when pred is false.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)

## Shapes

`pred` must be either scalar or the same shape as `true_value`.
`true_value` and `false_value` must have the same shape. Output has the
shape of `true_value`.

## StableHLO

Calls
[`stablehlo::hlo_select()`](https://r-xla.github.io/stablehlo/reference/hlo_select.html).

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
