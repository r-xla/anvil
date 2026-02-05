# Transform a graph to its gradient

Low-level graph transformation that appends the backward pass to a
traced
[`AnvilGraph`](https://r-xla.github.io/anvil/reference/AnvilGraph.md).
The function `f` represented by `graph` must return a single float
scalar. The resulting graph computes the gradients of that scalar with
respect to the inputs specified by `wrt`.

The backward rules are stored in `$rules[["backward"]]` of the
primitives.

This is the building block used by
[`gradient()`](https://r-xla.github.io/anvil/reference/gradient.md) and
[`value_and_gradient()`](https://r-xla.github.io/anvil/reference/value_and_gradient.md);
prefer those higher-level wrappers unless you need to operate on graphs
directly.

## Usage

``` r
transform_gradient(graph, wrt)
```

## Arguments

- graph:

  ([`AnvilGraph`](https://r-xla.github.io/anvil/reference/AnvilGraph.md))  
  The graph to transform. Must produce a single scalar float output.

- wrt:

  (`character`)  
  Names of the graph inputs to differentiate with respect to.

## Value

An [`AnvilGraph`](https://r-xla.github.io/anvil/reference/AnvilGraph.md)
whose outputs are the requested gradients.

## See also

[`gradient()`](https://r-xla.github.io/anvil/reference/gradient.md),
[`value_and_gradient()`](https://r-xla.github.io/anvil/reference/value_and_gradient.md)

## Examples

``` r
graph <- trace_fn(nvl_mul, list(nv_aten("f32", c()), nv_aten("f32", c())))
graph
#> <AnvilGraph>
#>   Inputs:
#>     %x1: f32[]
#>     %x2: f32[]
#>   Body:
#>     %1: f32[] = mul(%x1, %x2)
#>   Outputs:
#>     %1: f32[] 
transform_gradient(graph, "lhs")
#> <AnvilGraph>
#>   Inputs:
#>     %x1: f32[]
#>     %x2: f32[]
#>   Constants:
#>     %c1: f32[]
#>   Body:
#>     %1: f32[] = mul(%x1, %x2)
#>     %2: f32[] = mul(%c1, %x2)
#>   Outputs:
#>     %2: f32[] 
```
