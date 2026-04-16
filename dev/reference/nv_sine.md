# Sine

Element-wise sine. You can also use
[`sin()`](https://rdrr.io/r/base/Trig.html).

## Usage

``` r
nv_sine(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_sine()`](https://r-xla.github.io/anvil/dev/reference/nvl_sine.md)
for the underlying primitive.

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
jit_eval({
  x <- nv_array(c(0, pi / 2, pi))
  sin(x)
})
}
```
