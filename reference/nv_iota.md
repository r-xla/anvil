# Iota

Creates a tensor with values increasing along the specified dimension.

## Usage

``` r
nv_iota(dim, dtype, shape)
```

## Arguments

- dim:

  (`integer(1)`)  
  Dimension along which values increase.

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

## Value

[`tensorish`](tensorish.md)
