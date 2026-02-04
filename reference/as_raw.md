# Convert a tensor to a raw vector

Returns the underlying bytes of a tensor as a
[raw](https://rdrr.io/r/base/raw.html) vector.

## Usage

``` r
as_raw(x, ...)
```

## Arguments

- x:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  A tensor-like object.

- ...:

  Additional arguments passed to method:

  - `row_major` (`logical(1)`)  
    Whether to write the bytes in row-major order.

## Value

A [`raw`](https://rdrr.io/r/base/raw.html) vector.

## Details

This is implemented via the generic
[`tengen::as_raw()`](https://r-xla.github.io/tengen/reference/as_raw.html).

## Examples

``` r
x <- nv_tensor(1:4, shape = c(2, 2), dtype = "f32")
as_raw(x, row_major = TRUE)
#>  [1] 00 00 80 3f 00 00 40 40 00 00 00 40 00 00 80 40
as_raw(x, row_major = FALSE)
#>  [1] 00 00 80 3f 00 00 00 40 00 00 40 40 00 00 80 40
```
