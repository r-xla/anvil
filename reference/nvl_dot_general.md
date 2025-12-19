# Primitive Dot General

General dot product of two tensors.

## Usage

``` r
nvl_dot_general(lhs, rhs, contracting_dims, batching_dims)
```

## Arguments

- lhs, rhs:

  ([`tensorish`](tensorish.md))  
  Left and right operand.

- contracting_dims:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  Dimensions to contract.

- batching_dims:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  Batch dimensions.

## Value

[`tensorish`](tensorish.md)
