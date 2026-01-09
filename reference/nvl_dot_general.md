# Primitive Dot General

General dot product of two tensors.

## Usage

``` r
nvl_dot_general(lhs, rhs, contracting_dims, batching_dims)
```

## Arguments

- lhs, rhs:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Left and right operand.

- contracting_dims:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  Dimensions to contract.

- batching_dims:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  Batch dimensions.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
