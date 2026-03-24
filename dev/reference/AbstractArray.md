# Abstract Array Class

Representation of an abstract array type. During tracing, it is wrapped
in a
[`GraphNode`](https://r-xla.github.io/anvil/dev/reference/GraphNode.md)
held by a
[`GraphBox`](https://r-xla.github.io/anvil/dev/reference/GraphBox.md).
In the lowered
[`AnvilGraph`](https://r-xla.github.io/anvil/dev/reference/AnvilGraph.md)
it is also part of
[`GraphNode`](https://r-xla.github.io/anvil/dev/reference/GraphNode.md)s
representing the values in the program.

The base class represents an *unknown* value, but child classes exist
for:

- closed-over constants:
  [`ConcreteArray`](https://r-xla.github.io/anvil/dev/reference/ConcreteArray.md)

- scalar arrays arising from R literals:
  [`LiteralArray`](https://r-xla.github.io/anvil/dev/reference/LiteralArray.md)

- sequence patterns:
  [`IotaArray`](https://r-xla.github.io/anvil/dev/reference/IotaArray.md)

To convert a
[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)
value to an abstract array, use
[`to_abstract()`](https://r-xla.github.io/anvil/dev/reference/to_abstract.md).

## Usage

``` r
nv_aten(dtype, shape, ambiguous = FALSE)

AbstractArray(dtype, shape, ambiguous = FALSE)
```

## Arguments

- dtype:

  ([`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html)
  \| `character(1)`)  
  The data type of the array.

- shape:

  ([`stablehlo::Shape`](https://r-xla.github.io/stablehlo/reference/Shape.html)
  \| [`integer()`](https://rdrr.io/r/base/integer.html))  
  The shape of the array. Can be provided as an integer vector.

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the
  [`vignette("type-promotion")`](https://r-xla.github.io/anvil/dev/articles/type-promotion.md)
  for more details.

## Extractors

The following extractors are available on `AbstractArray` objects:

- [`dtype()`](https://r-xla.github.io/tengen/reference/dtype.html): Get
  the data type of the array.

- [`shape()`](https://r-xla.github.io/tengen/reference/shape.html): Get
  the shape (dimensions) of the array.

- [`ambiguous()`](https://r-xla.github.io/anvil/dev/reference/ambiguous.md):
  Get whether the dtype is ambiguous.

- [`ndims()`](https://r-xla.github.io/tengen/reference/ndims.html): Get
  the number of dimensions.

## See also

[LiteralArray](https://r-xla.github.io/anvil/dev/reference/LiteralArray.md),
[ConcreteArray](https://r-xla.github.io/anvil/dev/reference/ConcreteArray.md),
[IotaArray](https://r-xla.github.io/anvil/dev/reference/IotaArray.md),
[GraphValue](https://r-xla.github.io/anvil/dev/reference/GraphValue.md),
[`to_abstract()`](https://r-xla.github.io/anvil/dev/reference/to_abstract.md),
[GraphBox](https://r-xla.github.io/anvil/dev/reference/GraphBox.md)

## Examples

``` r
# -- Creating abstract arrays --
a <- AbstractArray("f32", c(2L, 3L))
a
#> AbstractArray(dtype=f32, shape=2x3) 
dtype(a)
#> <f32>
shape(a)
#> [1] 2 3
ambiguous(a)
#> [1] FALSE

# Shorthand
nv_aten("f32", c(2L, 3L))
#> AbstractArray(dtype=f32, shape=2x3) 

# How AbstractArrays appear in an AnvilGraph
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
#> AbstractArray(dtype=i32, shape=4) 
```
