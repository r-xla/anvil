# Exponential

Element-wise exponential. You can also use
[`exp()`](https://rdrr.io/r/base/Log.html).

## Usage

``` r
nv_exp(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_exp()`](https://r-xla.github.io/anvil/dev/reference/nvl_exp.md)
for the underlying primitive.

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
jit_eval({
  x <- nv_array(c(0, 1, 2))
  exp(x)
})
}
```
