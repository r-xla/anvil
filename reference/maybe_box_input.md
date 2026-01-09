# Maybe Box Input

This function is called onto the inputs of a function that is being
traced via
[`trace_fn()`](https://r-xla.github.io/anvil/reference/trace_fn.md). The
function boxes the provided value and registers it as an input to the
graph.

## Usage

``` r
maybe_box_input(x, desc, toplevel)
```

## Arguments

- x:

  (any)  
  The input to box.

- desc:

  ([`GraphDescriptor`](https://r-xla.github.io/anvil/reference/GraphDescriptor.md))  
  The descriptor of the graph.

- toplevel:

  (`logical(1)`)  
  Whether the function is being traced at the top level. If this is
  `TRUE`, inputs that are `AnvilTensor`s are treated as unknown, i.e. as
  `GraphValue` instead of `GraphLiteral` objects.
