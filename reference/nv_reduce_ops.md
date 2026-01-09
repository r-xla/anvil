# Reduction Operators

Reduce a tensor along specified dimensions.

## Usage

``` r
nv_reduce_sum(operand, dims, drop = TRUE)

nv_reduce_mean(operand, dims, drop = TRUE)

nv_reduce_prod(operand, dims, drop = TRUE)

nv_reduce_max(operand, dims, drop = TRUE)

nv_reduce_min(operand, dims, drop = TRUE)

nv_reduce_any(operand, dims, drop = TRUE)

nv_reduce_all(operand, dims, drop = TRUE)
```

## Arguments

- operand:

  ([`tensorish`](tensorish.md))  
  Operand.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions to reduce.

- drop:

  (`logical(1)`)  
  Whether to drop the reduced dimensions.

## Value

[`tensorish`](tensorish.md)
