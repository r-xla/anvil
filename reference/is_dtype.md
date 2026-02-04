# Check if an object is a TensorDataType

Tests whether `x` is a `TensorDataType` object.

## Usage

``` r
is_dtype(x)
```

## Arguments

- x:

  An object to test.

## Value

`TRUE` or `FALSE`.

## See also

[`as_dtype()`](https://r-xla.github.io/anvil/reference/as_dtype.md),
[`stablehlo::is_dtype()`](https://r-xla.github.io/stablehlo/reference/is_dtype.html)

## Examples

``` r
is_dtype("f32")
#> [1] FALSE
is_dtype(as_dtype("f32"))
#> [1] TRUE
```
