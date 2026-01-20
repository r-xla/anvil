# Concatenate

Concatenate a variadic number of tensors.

## Usage

``` r
nv_concatenate(..., dimension = NULL)
```

## Arguments

- ...:

  tensors

- dimension:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  The dimension to concatenate along to. Other dimensions must be the
  same. If this is `NULL` (default), it assumes all ranks are at most 1
  and the concatenation dimension is 1.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
