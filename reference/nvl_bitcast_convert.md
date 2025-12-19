# Primitive Bitcast Convert

Reinterprets tensor bits as a different dtype.

## Usage

``` r
nvl_bitcast_convert(operand, dtype)
```

## Arguments

- operand:

  ([`tensorish`](tensorish.md))  
  Operand.

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

## Value

[`tensorish`](tensorish.md)
