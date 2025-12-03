# Trace an R function into a Graph

Create a graph representation of an R function by tracing.

## Usage

``` r
trace_fn(f, args, desc = NULL)
```

## Arguments

- f:

  (`function`)  
  The function to trace_fn.

- args:

  (`list`)  
  The arguments to the function.

- desc:

  (`NULL` \| `GraphDescriptor`)  
  The descriptor to use for the graph.

## Value

([`Graph`](Graph.md))
