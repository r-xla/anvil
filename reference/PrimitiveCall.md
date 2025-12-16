# Primitive Call

Call of a primitive in a [`Graph`](Graph.md) Note that a primitive call
also be a call into another graph (`p_graph`).

## Usage

``` r
PrimitiveCall(
  primitive = Primitive(),
  params = list(),
  inputs = list(),
  outputs = list()
)
```

## Arguments

- primitive:

  (`Primitive`)  
  The function.

- params:

  (`list(<any>)`)  
  The (static) parameters of the function call.

- inputs:

  (`list(GraphValue)`)  
  The (tensor) inputs to the primitive.

- outputs:

  (`list(GraphValue)`)  
  The (tensor) outputs of the primitive.
