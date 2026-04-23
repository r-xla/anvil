# Primitive Call

Call of a primitive in an
[`AnvlGraph`](https://r-xla.github.io/anvl/dev/reference/AnvlGraph.md).

## Usage

``` r
PrimitiveCall(primitive, inputs, params, outputs)
```

## Arguments

- primitive:

  (`AnvlPrimitive`)  
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
