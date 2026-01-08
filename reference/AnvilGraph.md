# Graph of Primitive Calls

Computational graph consisting exclusively of primitive calls. This is a
mutable class.

## Usage

``` r
AnvilGraph(
  calls = list(),
  in_tree = NULL,
  out_tree = NULL,
  inputs = list(),
  outputs = list(),
  constants = list()
)
```

## Arguments

- calls:

  (`list(PrimitiveCall)`)  
  The primitive calls that make up the graph. This can also be another
  call into a graph when the primitive is a `p_call`.

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

- constants:

  (`list(GraphValue)`)  
  The constants of the graph.

## Value

(`AnvilGraph`)
