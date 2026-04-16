# Outer Product

Computes the outer product of two 1-D arrays.

## Usage

``` r
nv_outer(x, y)
```

## Arguments

- x, y:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  1-D arrays.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
A 2-D array of shape `(length(x), length(y))`.

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
jit_eval({
  x <- nv_array(c(1, 2, 3))
  y <- nv_array(c(4, 5))
  nv_outer(x, y)
})
}
```
