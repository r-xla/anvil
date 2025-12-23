# Primitive Iota

Creates a tensor with increasing values along a dimension.

## Usage

``` r
nvl_iota(dim, shape, dtype, start)
```

## Arguments

- dim:

  (`integer(1)`)  
  The dimension along which to generate increasing values.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

- start:

  (`numeric(1)`)  
  The value to start the sequence at.

## Value

[`tensorish`](tensorish.md)
