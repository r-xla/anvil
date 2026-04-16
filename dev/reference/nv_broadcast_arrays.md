# Broadcast Arrays to a Common Shape

Broadcasts arrays to a common shape using NumPy-style broadcasting
rules.

## Usage

``` r
nv_broadcast_arrays(...)
```

## Arguments

- ...:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrays to broadcast.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
List of arrays, all with the same shape.

## Broadcasting Rules

1.  If the arrays have different numbers of dimensions, prepend size-1
    dimensions to the shorter shape.

2.  For each dimension: if the sizes match, keep them; if one is 1,
    expand it to the other's size; otherwise raise an error.

## See also

[`nv_broadcast_scalars()`](https://r-xla.github.io/anvil/dev/reference/nv_broadcast_scalars.md),
[`nv_broadcast_to()`](https://r-xla.github.io/anvil/dev/reference/nv_broadcast_to.md)

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
jit_eval({
  x <- nv_array(matrix(1:6, nrow = 2))
  y <- nv_array(c(10, 20, 30))
  nv_broadcast_arrays(x, y)
})
}
```
