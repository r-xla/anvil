# Transform a graph to its gradient

Transform a graph to its gradient. This is a low-level function that
should usually not be used directly. Use [`gradient()`](gradient.md)
instead.

## Usage

``` r
transform_gradient(graph, wrt)
```

## Arguments

- graph:

  (`Graph`)  
  The graph to transform.

- wrt:

  (`character`)  
  The names of the variables to compute the gradient with respect to.

## Value

A [`Graph`](Graph.md) object.
