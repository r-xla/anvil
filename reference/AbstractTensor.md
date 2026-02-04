# Abstract Tensor Class

Representation of an abstract tensor type. During tracing, it is wrapped
in a [`GraphNode`](https://r-xla.github.io/anvil/reference/GraphNode.md)
held by a
[`GraphBox`](https://r-xla.github.io/anvil/reference/GraphBox.md). In
the lowered
[`AnvilGraph`](https://r-xla.github.io/anvil/reference/AnvilGraph.md) it
is also part of
[`GraphNode`](https://r-xla.github.io/anvil/reference/GraphNode.md)s
representing the values in the program.

The base class represents an *unknown* value, but child classes exist
for:

- closed-over constants:
  [`ConcreteTensor`](https://r-xla.github.io/anvil/reference/ConcreteTensor.md)

- scalar tensors arising from R literals:
  [`LiteralTensor`](https://r-xla.github.io/anvil/reference/LiteralTensor.md)

- sequence patterns:
  [`IotaTensor`](https://r-xla.github.io/anvil/reference/IotaTensor.md)

To convert a
[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
value to an abstract tensor, use
[`to_abstract()`](https://r-xla.github.io/anvil/reference/to_abstract.md).

## Usage

``` r
nv_aten(dtype, shape, ambiguous = FALSE)

AbstractTensor(dtype, shape, ambiguous = FALSE)
```

## Arguments

- dtype:

  ([`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html)
  \| `character(1)`)  
  The data type of the tensor.

- shape:

  ([`stablehlo::Shape`](https://r-xla.github.io/stablehlo/reference/Shape.html)
  \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  The shape of the tensor. Can be provided as an integer vector.

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the
  [`vignette("type-promotion")`](https://r-xla.github.io/anvil/articles/type-promotion.md)
  for more details.

## Extractors

The following extractors are available on `AbstractTensor` objects:

- [`dtype()`](https://r-xla.github.io/tengen/reference/dtype.html): Get
  the data type of the tensor.

- [`shape()`](https://r-xla.github.io/tengen/reference/shape.html): Get
  the shape (dimensions) of the tensor.

- [`ambiguous()`](https://r-xla.github.io/anvil/reference/ambiguous.md):
  Get whether the dtype is ambiguous.

- [`ndims()`](https://r-xla.github.io/tengen/reference/ndims.html): Get
  the number of dimensions.

## See also

[LiteralTensor](https://r-xla.github.io/anvil/reference/LiteralTensor.md),
[ConcreteTensor](https://r-xla.github.io/anvil/reference/ConcreteTensor.md),
[IotaTensor](https://r-xla.github.io/anvil/reference/IotaTensor.md),
[GraphValue](https://r-xla.github.io/anvil/reference/GraphValue.md),
[`to_abstract()`](https://r-xla.github.io/anvil/reference/to_abstract.md),
[GraphBox](https://r-xla.github.io/anvil/reference/GraphBox.md)

## Examples

``` r
# -- Creating abstract tensors --
a <- AbstractTensor("f32", c(2L, 3L))
a
#> AbstractTensor(dtype=f32, shape=2x3) 
dtype(a)
#> <f32>
shape(a)
#> [1] 2 3
ambiguous(a)
#> [1] FALSE

# Shorthand
nv_aten("f32", c(2L, 3L))
#> AbstractTensor(dtype=f32, shape=2x3) 

# How AbstractTensors appear in an AnvilGraph
graph <- trace_fn(function(x) x + 1, list(x = nv_aten("i32", 4L)))
graph
#> <AnvilGraph>
#>   Inputs:
#>     %x1: i32[4]
#>   Body:
#>     %1: f32?[4] = convert [dtype = f32, ambiguous = TRUE] (%x1)
#>     %2: f32?[4] = broadcast_in_dim [shape = 4, broadcast_dimensions = <any>] (1:f32?)
#>     %3: f32?[4] = add(%1, %2)
#>   Outputs:
#>     %3: f32?[4] 
graph$inputs[[1]]$aval
#> AbstractTensor(dtype=i32, shape=4) 
```
