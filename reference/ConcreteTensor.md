# Concrete Tensor Class

A
[`AbstractTensor`](https://r-xla.github.io/anvil/reference/AbstractTensor.md)
that also holds a reference to the actual tensor data. Used to represent
constants captured during tracing. Because it comes from a concrete
tensor, it's type is never ambiguous.

## Usage

``` r
ConcreteTensor(data)
```

## Arguments

- data:

  ([`AnvilTensor`](https://r-xla.github.io/anvil/reference/AnvilTensor.md))  
  The actual tensor data.

## See also

[AbstractTensor](https://r-xla.github.io/anvil/reference/AbstractTensor.md),
[LiteralTensor](https://r-xla.github.io/anvil/reference/LiteralTensor.md)
