# Concatenate

Concatenate a variadic amaount of tensors.

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

- operand:

  ([`nv_tensor`](nv_tensor.md))  
  Operand.

## Value

[`nv_tensor`](nv_tensor.md)
