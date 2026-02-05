# Lower a graph to StableHLO

Converts a traced
[`AnvilGraph`](https://r-xla.github.io/anvil/reference/AnvilGraph.md)
into the StableHLO intermediate representation (IR). Each graph
operation is translated to its corresponding StableHLO op. The result
can be serialized to MLIR text via
[`stablehlo::repr()`](https://r-xla.github.io/stablehlo/reference/repr.html)
and subsequently compiled to an XLA executable with
[`pjrt::pjrt_compile()`](https://r-xla.github.io/pjrt/reference/pjrt_compile.html).

The rules for translating to stablehlo are stored in
`$rules[["stablehlo"]]` of the primitives.

This is a low-level function; most users should use
[`jit()`](https://r-xla.github.io/anvil/reference/jit.md) or
[`xla()`](https://r-xla.github.io/anvil/reference/xla.md) instead.

## Usage

``` r
stablehlo(graph, constants_as_inputs = TRUE, env = NULL, donate = character())
```

## Arguments

- graph:

  ([`AnvilGraph`](https://r-xla.github.io/anvil/reference/AnvilGraph.md))  
  The graph to lower (e.g. produced by
  [`trace_fn()`](https://r-xla.github.io/anvil/reference/trace_fn.md)).

- constants_as_inputs:

  (`logical(1)`)  
  If `TRUE` (default), constants are registered as inputs to the
  StableHLO function so they can be passed in at execution time. If
  `FALSE`, they are not added as inputs. Set to `FALSE` for closures.
  Note that `GraphLiteral`s are always inlined into the StableHLO
  function.

- env:

  (`HloEnv` \| `NULL`)  
  Optional environment for reusing variable mappings across nested
  function lowerings (e.g. for higher-order primitives like `nv_while`).

- donate:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of the arguments whose buffers should be donated. Donated
  buffers can be aliased with outputs of the same type, enabling
  in-place operations.

## Value

A `list` of length 2:

- the
  [`stablehlo::Func`](https://r-xla.github.io/stablehlo/reference/Func.html)

- The list of
  [`GraphValue`](https://r-xla.github.io/anvil/reference/GraphValue.md)s
  holding
  [`ConcreteTensor`](https://r-xla.github.io/anvil/reference/ConcreteTensor.md)s.

## See also

[`trace_fn()`](https://r-xla.github.io/anvil/reference/trace_fn.md),
[`jit()`](https://r-xla.github.io/anvil/reference/jit.md),
[`xla()`](https://r-xla.github.io/anvil/reference/xla.md)

## Examples

``` r
x <- nv_tensor(c(1, 2))
graph <- trace_fn(function(y) y + x, list(y = nv_aten("f32", shape = c())))
graph
#> <AnvilGraph>
#>   Inputs:
#>     %x1: f32[]
#>   Constants:
#>     %c1: f32[2]
#>   Body:
#>     %1: f32[2] = broadcast_in_dim [shape = 2, broadcast_dimensions = <any>] (%x1)
#>     %2: f32[2] = add(%1, %c1)
#>   Outputs:
#>     %2: f32[2] 
stablehlo(graph)
#> [[1]]
#> func.func @main (%0: tensor<2xf32>, %1: tensor<f32>) -> tensor<2xf32> {
#> %2 = "stablehlo.broadcast_in_dim" (%1) {
#> broadcast_dimensions = array<i64>
#> }: (tensor<f32>) -> (tensor<2xf32>)
#> %3 = "stablehlo.add" (%2, %0): (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>)
#> "func.return"(%3): (tensor<2xf32>) -> ()
#> }
#> 
#> [[2]]
#> [[2]][[1]]
#> GraphValue(ConcreteTensor(f32, (2))) 
#> 
#> 
```
