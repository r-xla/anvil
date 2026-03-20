# Convert to a TensorDataType

Coerces a value to a `TensorDataType`. Accepts data type strings (e.g.
`"f32"`, `"i64"`, `"bool"`) or existing `TensorDataType` objects (they
are returned unchanged).

## Usage

``` r
as_dtype(x)
```

## Arguments

- x:

  A character string or `TensorDataType` to convert.

## Value

A `TensorDataType` object.

## Details

This is implemented via the generic
[`tengen::as_dtype()`](https://r-xla.github.io/tengen/reference/as_dtype.html).

## See also

[`is_dtype()`](https://r-xla.github.io/anvil/dev/reference/is_dtype.md),
[`tengen::as_dtype()`](https://r-xla.github.io/tengen/reference/as_dtype.html),
[`tengen::TensorDataType`](https://r-xla.github.io/tengen/reference/TensorDataType.html)

## Examples

``` r
as_dtype("f32")
#> <f32>
as_dtype("i32")
#> <i32>
```
