# Create a graph

Creates a new [`Graph`](Graph.md) which is afterwards accessible via
[`.current_descriptor()`](dot-current_descriptor.md). The graph is
automatically removed when exiting the current scope. After the graph is
either cleaned up automatically (by exiting the scope) or finalized, the
previously built graph is restored, i.e., accessible via
[`.current_descriptor()`](dot-current_descriptor.md).

## Usage

``` r
local_descriptor(..., envir = parent.frame())
```

## Arguments

- ...:

  (`any`)  
  Additional arguments to pass to the
  [`GraphDescriptor`](GraphDescriptor.md) constructor.

- envir:

  (`environment`)  
  Environment where exit handler will be registered for cleaning up the
  [`Graph`](Graph.md) if it was not returned yet.

## Value

A [`Graph`](Graph.md) object.
