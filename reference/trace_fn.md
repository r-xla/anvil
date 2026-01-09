# Trace an R function into a Graph

Create a graph representation of an R function by tracing.

## Usage

``` r
trace_fn(
  f,
  args = NULL,
  desc = NULL,
  toplevel = FALSE,
  flat_inputs = NULL,
  in_tree = NULL
)
```

## Arguments

- f:

  (`function`)  
  The function to trace_fn.

- args:

  (`list` of
  ([`AnvilTensor`](https://r-xla.github.io/anvil/reference/AnvilTensor.md)
  \|
  [`AbstractTensor`](https://r-xla.github.io/anvil/reference/AbstractTensor.md)))  
  The arguments to the function. Can be `NULL` if `flat_inputs` is
  provided.

- desc:

  (`NULL` \| `GraphDescriptor`)  
  The descriptor to use for the graph.

- toplevel:

  (`logical(1)`)  
  Whether the function is being traced at the top level. If this is
  `TRUE`, inputs that are `AnvilTensor`s are treated as unknown. If this
  is `FALSE` (default), `AnvilTensor`s are treated as constants.

- flat_inputs:

  (`NULL` \| `list`)  
  Pre-flattened inputs. If provided, `args` is ignored and `in_tree`
  must be provided.

- in_tree:

  (`NULL` \| `Node`)  
  The input tree structure. Required if `flat_inputs` is provided.

## Value

([`AnvilGraph`](https://r-xla.github.io/anvil/reference/AnvilGraph.md))
