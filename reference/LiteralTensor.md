# Literal Tensor Class

A [`ShapedTensor`](ShapedTensor.md) representing a tensor where the data
is a R scalar literal (e.g., `1L`, `2.5`, `TRUE`). Their type is
ambiguous, and they adapt (if possible) to the types of non-literal
tensors they interact with.

## Usage

``` r
LiteralTensor(data, shape, dtype = default_dtype(data))
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

## See also

[ShapedTensor](ShapedTensor.md), [ConcreteTensor](ConcreteTensor.md)
