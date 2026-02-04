# Get the shape of a tensor

Returns the shape of a tensor as an
[`integer()`](https://rdrr.io/r/base/integer.html) vector.

## Usage

``` r
shape(x, ...)
```

## Arguments

- x:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  A tensor-like object.

- ...:

  Additional arguments passed to methods (unused).

## Value

[`integer()`](https://rdrr.io/r/base/integer.html)

## Details

This is implemented via the generic
[`tengen::shape()`](https://r-xla.github.io/tengen/reference/shape.html).

## Examples

``` r
x <- nv_tensor(1:4, dtype = "f32")
shape(x)
#> [1] 4
```
