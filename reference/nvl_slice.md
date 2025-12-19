# Primitive Slice

Extracts a slice from a tensor.

## Usage

``` r
nvl_slice(operand, start_indices, limit_indices, strides)
```

## Arguments

- operand:

  ([`tensorish`](tensorish.md))  
  Operand.

- start_indices:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Start indices (1-based).

- limit_indices:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  End indices (exclusive).

- strides:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Step sizes.

## Value

[`tensorish`](tensorish.md)
