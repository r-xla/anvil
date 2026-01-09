# Concatenate

Concatenate a variadic number of tensors.

## Usage

``` r
nv_concatenate(..., dimension)
```

## Arguments

- ...:

  tensors

- dimension:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  The dimension to concatenate along to. Other dimensions must be the
  same.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
