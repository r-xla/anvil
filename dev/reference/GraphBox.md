# Graph Box

An [`AnvilBox`](https://r-xla.github.io/anvil/dev/reference/AnvilBox.md)
subclass that wraps a
[`GraphNode`](https://r-xla.github.io/anvil/dev/reference/GraphNode.md)
during graph construction (tracing). When a function is traced via
[`trace_fn()`](https://r-xla.github.io/anvil/dev/reference/trace_fn.md),
each intermediate tensor value is represented as a `GraphBox`. It also
contains an associated
[`GraphDescriptor`](https://r-xla.github.io/anvil/dev/reference/GraphDescriptor.md)
in which the node "lives".

## Usage

``` r
GraphBox(gnode, desc)
```

## Arguments

- gnode:

  ([`GraphNode`](https://r-xla.github.io/anvil/dev/reference/GraphNode.md))  
  The graph node – either a
  [`GraphValue`](https://r-xla.github.io/anvil/dev/reference/GraphValue.md)
  or a
  [`GraphLiteral`](https://r-xla.github.io/anvil/dev/reference/GraphLiteral.md).

- desc:

  ([`GraphDescriptor`](https://r-xla.github.io/anvil/dev/reference/GraphDescriptor.md))  
  The descriptor of the graph being built.

## Value

(`GraphBox`)

## Extractors

- [`dtype()`](https://r-xla.github.io/tengen/reference/dtype.html)

- [`shape()`](https://r-xla.github.io/tengen/reference/shape.html)

- [`ndims()`](https://r-xla.github.io/tengen/reference/ndims.html)

- [`ambiguous()`](https://r-xla.github.io/anvil/dev/reference/ambiguous.md)

## See also

[AnvilBox](https://r-xla.github.io/anvil/dev/reference/AnvilBox.md),
[DebugBox](https://r-xla.github.io/anvil/dev/reference/DebugBox.md),
[`trace_fn()`](https://r-xla.github.io/anvil/dev/reference/trace_fn.md),
[`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md)
