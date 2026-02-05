# Conditional Element Selection

Selects elements from `true_value` or `false_value` based on `pred`,
analogous to R's [`ifelse()`](https://rdrr.io/r/base/ifelse.html).

## Usage

``` r
nv_ifelse(pred, true_value, false_value)
```

## Arguments

- pred:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
  of boolean type)  
  Predicate tensor. Must be scalar or the same shape as `true_value`.

- true_value:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Values to return where `pred` is `TRUE`.

- false_value:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Values to return where `pred` is `FALSE`. Must have the same shape and
  data type as `true_value`.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as `true_value`.

## See also

[`nvl_ifelse()`](https://r-xla.github.io/anvil/reference/nvl_ifelse.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  pred <- nv_tensor(c(TRUE, FALSE, TRUE))
  nv_ifelse(pred, nv_tensor(c(1, 2, 3)), nv_tensor(c(4, 5, 6)))
})
#> AnvilTensor
#>  1
#>  5
#>  3
#> [ CPUf32{3} ] 
```
