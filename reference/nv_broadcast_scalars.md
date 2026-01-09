# Broadcast Scalars to Common Shape

Broadcast scalar tensors to match the shape of non-scalar tensors. All
non-scalar tensors must have the same shape.

## Usage

``` r
nv_broadcast_scalars(...)
```

## Arguments

- ...:

  ([`tensorish`](tensorish.md))  
  Tensors to broadcast. Scalars will be broadcast to the common
  non-scalar shape.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`tensorish`](tensorish.md))  
List of broadcasted tensors.
