# Add a Primitive Call to a Graph Descriptor

Add a primitive call to a graph descriptor.

## Usage

``` r
graph_desc_add(
  prim,
  args,
  params = list(),
  infer_fn,
  desc = NULL,
  debug_mode = NULL
)
```

## Arguments

- prim:

  ([`Primitive`](Primitive.md))  
  The primitive to add.

- args:

  (`list` of [`GraphNode`](GraphNode.md))  
  The arguments to the primitive.

- params:

  (`list`)  
  The parameters to the primitive.

- infer_fn:

  (`function`)  
  The inference function to use. Must output a list of
  [`AbstractTensor`](AbstractTensor.md)s.

- desc:

  ([`GraphDescriptor`](GraphDescriptor.md) \| `NULL`)  
  The graph descriptor to add the primitive call to. Uses the [current
  descriptor](dot-current_descriptor.md) if `NULL`.

- debug_mode:

  (`logical(1)`)  
  Whether to just perform abstract evaluation for debugging.

## Value

(`list` of `Box`)  
Either `GraphBox` objects or `DebugBox` objects, depending on
`debug_mode`.
