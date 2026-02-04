# Get the platform of a tensor or buffer

Returns the platform name (e.g. `"cpu"`, `"cuda"`) identifying the
compute backend.

## Usage

``` r
platform(x, ...)
```

## Arguments

- x:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  A tensor-like object.

- ...:

  Additional arguments passed to methods (unused).

## Value

`character(1)`

## Details

Implemented via the generic
[`pjrt::platform()`](https://r-xla.github.io/pjrt/reference/platform.html).

## See also

[`pjrt::platform()`](https://r-xla.github.io/pjrt/reference/platform.html)

## Examples

``` r
x <- nv_tensor(1:4, dtype = "f32")
platform(x)
#> [1] "cpu"
```
