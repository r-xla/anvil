# Primitive Broadcast

Broadcasts a tensor to a new shape.

## Usage

``` r
nvl_broadcast_in_dim(operand, shape_out, broadcast_dimensions)
```

## Arguments

- operand:

  ([`tensorish`](tensorish.md))  
  Operand.

- shape_out:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Target shape.

- broadcast_dimensions:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimension mapping.

## Value

[`tensorish`](tensorish.md)
