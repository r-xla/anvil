# Abstract Tensor Class

Abstract representation of a tensor with a (possibly ambiguous) dtype
and shape, but no concrete data. Used during tracing to represent tensor
metadata without actual values.

## Usage

``` r
AbstractTensor(dtype, shape, ambiguous = FALSE)
```

## Arguments

- dtype:

  ([`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  The data type of the tensor.

- shape:

  ([`stablehlo::Shape`](https://r-xla.github.io/stablehlo/reference/Shape.html)
  \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  The shape of the tensor. Can be provided as an integer vector.

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous usually arise from R literals
  (e.g., `1L`, `1.0`) and follow special promotion rules. Only `f32`,
  `i32` are ambiguous during tracing.

## Details

Two tensors are considered equal (`==`) if they have the same dtype and
shape, ignoring ambiguity.

## See also

[ConcreteTensor](ConcreteTensor.md), [LiteralTensor](LiteralTensor.md),
[`to_abstract()`](to_abstract.md)
