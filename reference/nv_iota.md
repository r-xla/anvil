# Iota

Create a tensor with increasing values along a dimension.

Creates a tensor with values increasing along the specified dimension.

## Usage

``` r
nv_iota(iota_dimension, dtype, shape)

nv_iota(iota_dimension, dtype, shape)
```

## Arguments

- iota_dimension:

  (`integer(1)`)  
  Dimension along which values increase (1-indexed).

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- dim:

  (`integer(1)`)  
  The dimension along which to generate increasing values.

- start:

  (`numeric(1)`)  
  The value to start the sequence at (default 1).

## Value

[`tensorish`](tensorish.md)
