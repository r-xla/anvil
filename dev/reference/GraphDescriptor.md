# Graph Descriptor

Descriptor of an
[`AnvilGraph`](https://r-xla.github.io/anvil/dev/reference/AnvilGraph.md).
This is a mutable class.

## Usage

``` r
GraphDescriptor(
  calls = list(),
  tensor_to_gval = NULL,
  gval_to_box = NULL,
  constants = list(),
  in_tree = NULL,
  out_tree = NULL,
  inputs = list(),
  outputs = list(),
  is_static_flat = NULL,
  static_args_flat = NULL,
  devices = character(),
  backend = NULL
)
```

## Arguments

- calls:

  (`list(PrimitiveCall)`)  
  The primitive calls that make up the graph.

- tensor_to_gval:

  (`hashtab`)  
  Mapping: `AnvilArray` -\> `GraphValue`

- gval_to_box:

  (`hashtab`)  
  Mapping: `GraphValue` -\> `GraphBox`

- constants:

  (`list(GraphValue)`)  
  The constants of the graph.

- in_tree:

  (`NULL | Node`)  
  The tree of inputs. May contain leaves for both array inputs and
  static (non-array) arguments. Only the array leaves correspond to
  entries in `inputs`; use `is_static_flat` to distinguish them.

- out_tree:

  (`NULL | Node`)  
  The tree of outputs.

- inputs:

  (`list(GraphValue)`)  
  The inputs to the graph (array arguments only).

- outputs:

  (`list(GraphValue)`)  
  The outputs of the graph.

- is_static_flat:

  (`NULL | logical()`)  
  Boolean mask indicating which flat positions in `in_tree` are static
  (non-array) args. `NULL` when all args are array inputs.

- static_args_flat:

  (`NULL | list()`)  
  Flattened traced values for the static arguments indicated by
  `is_static_flat`.

- devices:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Device platforms encountered during tracing (e.g. `"cpu"`, `"cuda"`).
  Populated automatically as arrays are registered.

- backend:

  (`NULL` \| `"xla"` \| `"quickr"`)  
  Backend associated with this graph descriptor.

## Value

(`GraphDescriptor`)
