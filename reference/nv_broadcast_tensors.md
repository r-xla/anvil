# Broadcast Tensors to a Common Shape

Broadcast tensors to a common shape.

## Usage

``` r
nv_broadcast_tensors(...)
```

## Arguments

- ...:

  ([`tensorish`](tensorish.md))  
  Tensors to broadcast.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`tensorish`](tensorish.md))

## Broadcasting Rules

We follow the standard NumPy broadcasting rules:

1.  If the tensors have different numbers of dimensions, prepend 1s to
    the shape of the smaller tensor.

2.  For each dimension, if:

    - the sizes are the same, do nothing.

    - one of the tensors has size 1, expand it to the corresponding size
      of the other tensor.

    - the sizes are different and neither is 1, raise an error.
