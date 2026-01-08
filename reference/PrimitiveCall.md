# Primitive Call

Call of a primitive in a [`Graph`](Graph.md) Note that a primitive call
also be a call into another graph (`p_graph`).

## Usage

``` r
PrimitiveCall(primitive, inputs, params, outputs)
```

## Arguments

- primitive:

  (`Primitive`)  
  The function.

- inputs:

  (`list(GraphValue)`)  
  The (tensor) inputs to the primitive.

- params:

  (`list(<any>)`)  
  The (static) parameters of the function call.

- outputs:

  (`list(GraphValue)`)  
  The (tensor) outputs of the primitive.

## Value

(`PrimitiveCall`)
