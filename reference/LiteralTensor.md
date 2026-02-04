# Literal Tensor Class

An
[`AbstractTensor`](https://r-xla.github.io/anvil/reference/AbstractTensor.md)
where all elements have the same constant value. This either arises when
using literals in traced code (e.g. `x + 1`) or when using
[`nv_fill()`](https://r-xla.github.io/anvil/reference/nv_fill.md) to
create a constant.

## Usage

``` r
LiteralTensor(data, shape, dtype = default_dtype(data), ambiguous)
```

## Arguments

- data:

  (`double(1)` \| `integer(1)` \| `logical(1)` \|
  [`AnvilTensor`](https://r-xla.github.io/anvil/reference/AnvilTensor.md))  
  The scalar value or scalarish AnvilTensor (contains 1 element).

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
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the
  [`vignette("type-promotion")`](https://r-xla.github.io/anvil/articles/type-promotion.md)
  for more details.

## Type Ambiguity

When arising from R literals, the resulting `LiteralTensor` is ambiguous
because no type information was available. See the
[`vignette("type-promotion")`](https://r-xla.github.io/anvil/articles/type-promotion.md)
for more details.

## Lowering

`LiteralTensor`s become constants inlined into the stableHLO program.
I.e., they lower to
[`stablehlo::hlo_tensor()`](https://r-xla.github.io/stablehlo/reference/hlo_constant.html).

## Examples

``` r
x <- LiteralTensor(1L, shape = integer(), ambiguous = TRUE)
x
#> LiteralTensor(1, i32?, ()) 
ambiguous(x)
#> [1] TRUE
shape(x)
#> integer(0)
ndims(x)
#> [1] 0
dtype(x)
#> <i32>
# How it appears during tracing:
# 1. via R literals
graph <- trace_fn(function() 1, list())
graph
#> <AnvilGraph>
#>   Inputs: (none)
#>   Body: (empty)
#>   Outputs:
#>     1:f32? 
graph$outputs[[1]]$aval
#> LiteralTensor(1, f32?, ()) 
# 2. via nv_fill()
graph <- trace_fn(function() nv_fill(2L, shape = c(2, 2)), list())
graph
#> <AnvilGraph>
#>   Inputs: (none)
#>   Body:
#>     %1: i32[2, 2] = fill [value = 2, dtype = i32, shape = c(2, 2), ambiguous = FALSE] ()
#>   Outputs:
#>     %1: i32[2, 2] 
graph$outputs[[1]]$aval
#> AbstractTensor(dtype=i32, shape=2x2) 
```
