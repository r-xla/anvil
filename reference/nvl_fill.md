# Primitive Fill

Creates a tensor filled with a scalar value.

## Usage

``` r
nvl_fill(value, shape, dtype)
```

## Arguments

- value:

  (`numeric(1)`)  
  Scalar value.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

## Value

[`tensorish`](tensorish.md)
