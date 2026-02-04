# Convert to a TensorDataType

Coerces a value to a `TensorDataType`. Accepts data type strings (e.g.
`"f32"`, `"i64"`, `"i1"`) or existing `TensorDataType` objects (they are
returned unchanged).

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
[`stablehlo::as_dtype()`](https://r-xla.github.io/stablehlo/reference/as_dtype.html).

## See also

[`is_dtype()`](https://r-xla.github.io/anvil/reference/is_dtype.md),
[`stablehlo::as_dtype()`](https://r-xla.github.io/stablehlo/reference/as_dtype.html),
[`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html)

## Examples

``` r
as_dtype("f32")
#> <f32>
as_dtype("i32")
#> <i32>
```
