# Conditional Element Selection

Selects elements from `true_value` or `false_value` based on `pred`,
analogous to R's [`ifelse()`](https://rdrr.io/r/base/ifelse.html).

## Usage

``` r
nv_ifelse(pred, true_value, false_value)
```

## Arguments

- pred:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)
  of boolean type)  
  Predicate array. Must be scalar or the same shape as `true_value`.

- true_value:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Values to return where `pred` is `TRUE`.

- false_value:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Values to return where `pred` is `FALSE`. Must have the same shape and
  data type as `true_value`.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as `true_value`.

## See also

[`nvl_ifelse()`](https://r-xla.github.io/anvil/dev/reference/nvl_ifelse.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  pred <- nv_array(c(TRUE, FALSE, TRUE))
  nv_ifelse(pred, nv_array(c(1, 2, 3)), nv_array(c(4, 5, 6)))
})
#> AnvilArray
#>  1
#>  5
#>  3
#> [ CPUf32{3} ] 
```
