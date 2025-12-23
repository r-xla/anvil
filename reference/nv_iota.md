# Iota

Create a tensor with increasing values along a dimension.

## Usage

``` r
nv_iota(dim, shape, dtype = "i32", start = 1)
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
  The value to start the sequence at (default 1).
