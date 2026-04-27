# Add a Primitive Call to a Graph Descriptor

Add a primitive call to a graph descriptor. Inside a primitive body
created with
[`new_primitive()`](https://r-xla.github.io/anvl/reference/new_primitive.md),
pass the lexically-bound `self` as the primitive argument.

## Usage

``` r
graph_desc_add(primitive, args, params = list(), infer_fn, desc = NULL)
```

## Arguments

- primitive:

  ([`AnvlPrimitive`](https://r-xla.github.io/anvl/reference/AnvlPrimitive.md)
  \| `JitPrimitive`)  
  The primitive the call is for. A `JitPrimitive` is accepted and
  unwrapped to its underlying `AnvlPrimitive` metadata.

- args:

  (`list` of
  [`GraphNode`](https://r-xla.github.io/anvl/reference/GraphNode.md))  
  The arguments to the primitive.

- params:

  (`list`)  
  The parameters to the primitive.

- infer_fn:

  (`function`)  
  The inference function to use. Must output a list of
  [`AbstractArray`](https://r-xla.github.io/anvl/reference/AbstractArray.md)s.

- desc:

  ([`GraphDescriptor`](https://r-xla.github.io/anvl/reference/GraphDescriptor.md)
  \| `NULL`)  
  The graph descriptor to add the primitive call to. Uses the [current
  descriptor](https://r-xla.github.io/anvl/reference/dot-current_descriptor.md)
  if `NULL`.

## Value

(`list` of
[`GraphBox`](https://r-xla.github.io/anvl/reference/GraphBox.md))
