# Logistic (Sigmoid)

Element-wise logistic sigmoid: `1 / (1 + exp(-x))`.

## Usage

``` r
nv_logistic(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_logistic()`](https://r-xla.github.io/anvil/dev/reference/nvl_logistic.md)
for the underlying primitive.

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
jit_eval({
  x <- nv_array(c(-2, 0, 2))
  nv_logistic(x)
})
}
```
