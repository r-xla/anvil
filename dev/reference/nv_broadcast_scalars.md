# Broadcast Scalars to Common Shape

Broadcast scalar arrays to match the shape of non-scalar arrays. All
non-scalar arrays must have the same shape.

## Usage

``` r
nv_broadcast_scalars(...)
```

## Arguments

- ...:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrays to broadcast. Scalars will be broadcast to the common
  non-scalar shape.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
List of broadcasted arrays.

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
jit_eval({
  x <- nv_array(c(1, 2, 3))
  # scalar 1 is broadcast to shape [3]
  nv_broadcast_scalars(x, 1)
})
}
```
