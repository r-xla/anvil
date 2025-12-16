# Broadcast Scalars to Common Shape

Broadcast scalar tensors to match the shape of non-scalar tensors. All
non-scalar tensors must have the same shape.

## Usage

``` r
nv_broadcast_scalars(...)
```

## Arguments

- ...:

  ([`nv_tensor`](nv_tensor.md))  
  Tensors to broadcast. Scalars will be broadcast to the common
  non-scalar shape.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`nv_tensor`](nv_tensor.md))
