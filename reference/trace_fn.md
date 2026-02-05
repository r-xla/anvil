# Trace an R function into a Graph

Executes `f` with abstract tensor arguments and records every primitive
operation into an
[`AnvilGraph`](https://r-xla.github.io/anvil/reference/AnvilGraph.md).

The resulting graph can be lowered to StableHLO (via
[`stablehlo()`](https://r-xla.github.io/anvil/reference/stablehlo.md))
or transformed (e.g. via
[`transform_gradient()`](https://r-xla.github.io/anvil/reference/transform_gradient.md)).

## Usage

``` r
trace_fn(
  f,
  args = NULL,
  desc = NULL,
  toplevel = FALSE,
  lit_to_tensor = FALSE,
  args_flat = NULL,
  in_tree = NULL
)
```

## Arguments

- f:

  (`function`)  
  The function to trace. Must not be a `JitFunction` (i.e. already
  jitted).

- args:

  (`list` of
  ([`AnvilTensor`](https://r-xla.github.io/anvil/reference/AnvilTensor.md)
  \|
  [`AbstractTensor`](https://r-xla.github.io/anvil/reference/AbstractTensor.md)))  
  The (unflattened) arguments to the function. Mutually exclusive with
  the `args_flat`/`in_tree` pair.

- desc:

  (`NULL` \| `GraphDescriptor`)  
  Optional descriptor. When `NULL` (default), a new descriptor is
  created.

- toplevel:

  (`logical(1)`)  
  If `TRUE`, concrete
  [`AnvilTensor`](https://r-xla.github.io/anvil/reference/AnvilTensor.md)
  inputs are treated as unknown (traced) values. If `FALSE` (default),
  they are treated as known constants.

- lit_to_tensor:

  (`logical(1)`)  
  Whether to convert literal inputs to tensors. Used internally by
  higher-order primitives such as `nv_if` and `nv_while`.

- args_flat:

  (`list`)  
  Flattened arguments. Must be accompanied by `in_tree`.

- in_tree:

  (`Node`)  
  Tree structure describing how `args_flat` maps back to `f`'s
  arguments.

## Value

An [`AnvilGraph`](https://r-xla.github.io/anvil/reference/AnvilGraph.md)
containing the traced operations.

## See also

[`stablehlo()`](https://r-xla.github.io/anvil/reference/stablehlo.md) to
lower the graph,
[`jit()`](https://r-xla.github.io/anvil/reference/jit.md) /
[`xla()`](https://r-xla.github.io/anvil/reference/xla.md) for end-to-end
compilation.

## Examples

``` r
graph <- trace_fn(function(x, y) x + y,
  args = list(x = nv_tensor(1, dtype = "f32"), y = nv_tensor(2, dtype = "f32"))
)
graph
#> <AnvilGraph>
#>   Inputs:
#>     %x1: f32[1]
#>     %x2: f32[1]
#>   Body:
#>     %1: f32[1] = add(%x1, %x2)
#>   Outputs:
#>     %1: f32[1] 
```
