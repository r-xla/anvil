# Reshape

Reshape a tensor. Note that row-major order is used, which differs from
R's column-major order.

## Usage

``` r
nv_reshape(operand, shape)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  The new shape.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
