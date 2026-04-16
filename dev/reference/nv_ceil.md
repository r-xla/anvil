# Ceiling

Element-wise ceiling (round toward positive infinity). You can also use
[`ceiling()`](https://rdrr.io/r/base/Round.html).

## Usage

``` r
nv_ceil(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_ceil()`](https://r-xla.github.io/anvil/dev/reference/nvl_ceil.md)
for the underlying primitive.

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
jit_eval({
  x <- nv_array(c(1.2, 2.7, -1.5))
  ceiling(x)
})
}
```
