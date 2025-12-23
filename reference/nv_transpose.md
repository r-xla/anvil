# Transpose

Transpose a tensor.

## Usage

``` r
nv_transpose(x, permutation = NULL)

# S3 method for class '`anvil::Box`'
t(x)
```

## Arguments

- x:

  ([`nv_tensor`](nv_tensor.md))

- permutation:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULl`)  
  Permutation of dimensions. If `NULL` (default), reverses the
  dimensions.

## Value

[`nv_tensor`](nv_tensor.md)
