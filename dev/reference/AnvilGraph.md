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
  constants = list(),
  is_static_flat = NULL
)
```

## Arguments

- calls:

  (`list(PrimitiveCall)`)  
  The primitive calls that make up the graph. This can also be another
  call into a graph when the primitive is a `p_call`.

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

- constants:

  (`list(GraphValue)`)  
  The constants of the graph.

- is_static_flat:

  (`NULL | logical()`)  
  Boolean mask indicating which flat positions in `in_tree` are static
  (non-tensor) args. `NULL` when all args are tensor inputs.

## Value

(`AnvilGraph`)
