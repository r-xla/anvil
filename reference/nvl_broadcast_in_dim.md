# Primitive Broadcast

Broadcasts a tensor to a new shape.

## Usage

``` r
nvl_broadcast_in_dim(operand, shape, broadcast_dimensions)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Target shape.

- broadcast_dimensions:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimension mapping.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
