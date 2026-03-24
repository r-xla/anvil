# Convert to an R array

Transfers array data to R and returns it as an R
[`array`](https://rdrr.io/r/base/array.html). Only in the case of
scalars is the result a vector of length 1, as R `arrays` cannot have 0
dimensions.

## Usage

``` r
as_array(x, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  An array-like object.

- ...:

  Additional arguments passed to methods (unused).

## Value

An R [`array`](https://rdrr.io/r/base/array.html) or `vector` of length
1.

## Details

This is implemented via the generic
[`tengen::as_array()`](https://r-xla.github.io/tengen/reference/as_array.html).

## Examples

``` r
x <- nv_array(1:4, dtype = "f32")
as_array(x)
#> [1] 1 2 3 4
y <- nv_scalar(1L)
# R arrays can't have 0 dimensions:
as_array(y)
#> [1] 1
```
