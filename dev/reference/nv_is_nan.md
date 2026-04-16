# Is NaN

Element-wise check if values are NaN. You can also use
[`is.nan()`](https://rdrr.io/r/base/is.finite.html).

## Usage

``` r
nv_is_nan(operand)

# S3 method for class 'AnvilBox'
is.nan(x)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

- x:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape as the input and boolean data type.

## See also

[`nv_is_finite()`](https://r-xla.github.io/anvil/dev/reference/nv_is_finite.md),
[`nv_is_infinite()`](https://r-xla.github.io/anvil/dev/reference/nv_is_infinite.md)

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
jit_eval({
  x <- nv_array(c(1, NaN, Inf, -Inf, 0))
  nv_is_nan(x)
})
}
```
