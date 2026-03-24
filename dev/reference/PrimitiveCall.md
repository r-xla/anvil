# Primitive Call

Call of a primitive in an
[`AnvilGraph`](https://r-xla.github.io/anvil/dev/reference/AnvilGraph.md)
Note that a primitive call also be a call into another graph
(`p_graph`).

## Usage

``` r
PrimitiveCall(primitive, inputs, params, outputs)
```

## Arguments

- primitive:

  (`AnvilPrimitive`)  
  The function.

- inputs:

  (`list(GraphValue)`)  
  The (array) inputs to the primitive.

- params:

  (`list(<any>)`)  
  The (static) parameters of the function call.

- outputs:

  (`list(GraphValue)`)  
  The (array) outputs of the primitive.

## Value

(`PrimitiveCall`)
