# Get the number of dimensions of a tensor

Returns the number of dimensions (sometimes also refered to as rank) of
a tensor. Equivalent to `length(shape(x))`.

## Usage

``` r
ndims(x)
```

## Arguments

- x:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  A tensor-like object.

## Value

`integer(1)`

## See also

[`tengen::ndims()`](https://r-xla.github.io/tengen/reference/ndims.html)

## Examples

``` r
x <- nv_tensor(1:4, dtype = "f32")
ndims(x)
#> [1] 1
```
