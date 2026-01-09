# Primitive Sum Reduction

Sums tensor elements along dimensions.

## Usage

``` r
nvl_reduce_sum(operand, dims, drop = TRUE)
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
  Whether to drop reduced dimensions.

## Value

[`tensorish`](tensorish.md)
