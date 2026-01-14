# Trace an R function into a Graph

Create a graph representation of an R function by tracing.

## Usage

``` r
trace_fn(
  f,
  args = NULL,
  desc = NULL,
  toplevel = FALSE,
  tensorish_args = FALSE,
  args_flat = NULL,
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
  The (unflattened) arguments to the function.

- desc:

  (`NULL` \| `GraphDescriptor`)  
  The descriptor to use for the graph.

- toplevel:

  (`logical(1)`)  
  Whether the function is being traced at the top level. If this is
  `TRUE`, inputs that are `AnvilTensor`s are treated as unknown. If this
  is `FALSE` (default), `AnvilTensor`s are treated as constants.

- tensorish_args:

  (`logical(1)`)  
  Whether the arguments are all tensorish. If this is `TRUE`, we convert
  R literals to scalar tensors.

- args_flat:

  (`list`)  
  The flattened arguments. Also requires passing `in_tree`.

- in_tree:

  (`Node`)  
  The tree structure of the arguments.

## Value

([`AnvilGraph`](https://r-xla.github.io/anvil/reference/AnvilGraph.md))
