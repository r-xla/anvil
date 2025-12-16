# Literal Tensor Class

A [`AbstractTensor`](AbstractTensor.md) representing a tensor where the
data is a R scalar literal (e.g., `1L`, `2.5`). Usually, their type is
ambiguous, unless created via [`nv_fill`](nv_fill.md).

## Usage

``` r
LiteralTensor(data, shape, dtype = default_dtype(data), ambiguous)
```

## Arguments

- data:

  (`numeric(1)` \| `integer(1)` \| `logical(1)`)  
  The scalar value.

- shape:

  ([`stablehlo::Shape`](https://r-xla.github.io/stablehlo/reference/Shape.html)
  \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  The shape of the tensor.

- dtype:

  ([`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  The data type. Defaults to `f32` for numeric, `i32` for integer, `i1`
  for logical.

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous usually arise from R literals
  (e.g., `1L`, `1.0`) and follow special promotion rules. Only `f32` and
  `i32` can be ambiguous during tracing.

## See also

[AbstractTensor](AbstractTensor.md), [ConcreteTensor](ConcreteTensor.md)
