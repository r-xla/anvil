# Get the number of dimensions of an array

Returns the number of dimensions (sometimes also refered to as rank) of
an array. Equivalent to `length(shape(x))`.

## Usage

``` r
ndims(x)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  An array-like object.

## Value

`integer(1)`

## See also

[`tengen::ndims()`](https://r-xla.github.io/tengen/reference/ndims.html)

## Examples

``` r
x <- nv_array(1:4, dtype = "f32")
ndims(x)
#> [1] 1
```
