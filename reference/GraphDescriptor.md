# Graph Descriptor

Descriptor of a [`Graph`](Graph.md).

## Usage

``` r
GraphDescriptor(
  calls = list(),
  tensor_to_gval = hashtab(),
  gval_to_box = hashtab(),
  constants = list(),
  in_tree = NULL,
  out_tree = NULL,
  inputs = list(),
  outputs = list()
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
  The tree of inputs.

- out_tree:

  (`NULL | Node`)  
  The tree of outputs.

- inputs:

  (`list(GraphValue)`)  
  The inputs to the graph.

- outputs:

  (`list(GraphValue)`)  
  The outputs of the graph.
