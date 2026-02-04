# Graph Descriptor

Descriptor of an
[`AnvilGraph`](https://r-xla.github.io/anvil/reference/AnvilGraph.md).
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
  is_static_flat = NULL
)
```

## Arguments

- calls:

  (`list(PrimitiveCall)`)  
  The primitive calls that make up the graph.

- tensor_to_gval:

  (`hashtab`)  
  Mapping: `AnvilTensor` -\> `GraphValue`

- gval_to_box:

  (`hashtab`)  
  Mapping: `GraphValue` -\> `GraphBox`

- constants:

  (`list(GraphValue)`)  
  The constants of the graph.

- in_tree:

  (`NULL | Node`)  
  The tree of inputs. May contain leaves for both tensor inputs and
  static (non-tensor) arguments. Only the tensor leaves correspond to
  entries in `inputs`; use `is_static_flat` to distinguish them.

- out_tree:

  (`NULL | Node`)  
  The tree of outputs.

- inputs:

  (`list(GraphValue)`)  
  The inputs to the graph (tensor arguments only).

- outputs:

  (`list(GraphValue)`)  
  The outputs of the graph.

- is_static_flat:

  (`NULL | logical()`)  
  Boolean mask indicating which flat positions in `in_tree` are static
  (non-tensor) args. `NULL` when all args are tensor inputs.

## Value

(`GraphDescriptor`)
