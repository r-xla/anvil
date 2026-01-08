# Trace an R function into a Graph

Create a graph representation of an R function by tracing.

## Usage

``` r
trace_fn(f, args, desc = NULL, toplevel = FALSE)
```

## Arguments

- f:

  (`function`)  
  The function to trace_fn.

- args:

  (`list` of ([`AnvilTensor`](nv_tensor.md) \|
  [`AbstractTensor`](AbstractTensor.md)))  
  The arguments to the function.

- desc:

  (`NULL` \| `GraphDescriptor`)  
  The descriptor to use for the graph.

- toplevel:

  (`logical(1)`)  
  Whether the function is being traced at the top level. If this is
  `TRUE`, inputs that are `AnvilTensor`s are treated as unknown. If this
  is `FALSE` (default), `AnvilTensor`s are treated as constants.

## Value

([`Graph`](Graph.md))
