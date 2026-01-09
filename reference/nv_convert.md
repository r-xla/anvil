# Convert Tensor to Different Data Type

Convert a tensor to a different data type.

## Usage

``` r
nv_convert(operand, dtype)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
