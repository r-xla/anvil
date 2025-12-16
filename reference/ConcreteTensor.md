# Concrete Tensor Class

A [`AbstractTensor`](AbstractTensor.md) that also holds a reference to
the actual tensor data. Used to represent constants captured during
tracing. Because it comes from a concrete tensor, it's type is never
ambiguous.

## Usage

``` r
ConcreteTensor(data)
```

## Arguments

- data:

  ([`AnvilTensor`](nv_tensor.md))  
  The actual tensor data.

## See also

[AbstractTensor](AbstractTensor.md), [LiteralTensor](LiteralTensor.md)
