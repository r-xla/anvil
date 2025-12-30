# Primitive Pad

Pads a tensor with a given padding value.

## Usage

``` r
nvl_pad(
  operand,
  padding_value,
  edge_padding_low,
  edge_padding_high,
  interior_padding
)
```

## Arguments

- operand:

  ([`tensorish`](tensorish.md))  
  Operand.

- padding_value:

  ([`tensorish`](tensorish.md))  
  Scalar value to use for padding.

- edge_padding_low:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Amount of padding to add at the start of each dimension.

- edge_padding_high:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Amount of padding to add at the end of each dimension.

- interior_padding:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Amount of padding to add between elements in each dimension.

## Value

[`tensorish`](tensorish.md)
