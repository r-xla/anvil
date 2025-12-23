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
  Whether the result type is ambiguous.

## Value

[`tensorish`](tensorish.md)
