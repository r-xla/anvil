# Iota Tensor Class

An
[`AbstractTensor`](https://r-xla.github.io/anvil/reference/AbstractTensor.md)
representing a tensor where the data is a sequence of integers.

## Usage

``` r
IotaTensor(shape, dtype, dimension, start = 1L, ambiguous = FALSE)
```

## Arguments

- shape:

  ([`stablehlo::Shape`](https://r-xla.github.io/stablehlo/reference/Shape.html)
  \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  The shape of the tensor.

- dtype:

  ([`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  The data type.

- dimension:

  (`integer(1)`)  
  The dimension along which values increase.

- start:

  (`integer(1)`)  
  The starting value.

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the vignette "Type Promotion" for more details.
