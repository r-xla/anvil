# Primitive Convert

Converts tensor to a different dtype.

## Usage

``` r
nvl_convert(operand, dtype, ambiguous = FALSE)
```

## Arguments

- operand:

  ([`tensorish`](tensorish.md))  
  Operand.

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the vignette "Type Promotion" for more details.

## Value

[`tensorish`](tensorish.md)
