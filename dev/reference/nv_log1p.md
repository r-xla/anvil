# Log Plus One

Element-wise `log(1 + x)`, more accurate for small `x`.

## Usage

``` r
nv_log1p(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_log1p()`](https://r-xla.github.io/anvil/dev/reference/nvl_log1p.md)
for the underlying primitive.

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
jit_eval({
  x <- nv_array(c(0, 0.001, 1))
  nv_log1p(x)
})
}
```
