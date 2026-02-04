# Get the data type of a tensor

Returns the data type of a tensor (e.g. `f32`, `i64`).

## Usage

``` r
dtype(x, ...)
```

## Arguments

- x:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  A tensor-like object.

- ...:

  Additional arguments passed to methods (unused).

## Value

A
[`TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html).

## Details

This is implemented via the generic
[`tengen::dtype()`](https://r-xla.github.io/tengen/reference/dtype.html).

## See also

[`tengen::dtype()`](https://r-xla.github.io/tengen/reference/dtype.html)

## Examples

``` r
x <- nv_tensor(1:4, dtype = "f32")
dtype(x)
#> <f32>
```
