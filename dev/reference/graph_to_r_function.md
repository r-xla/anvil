# Convert an AnvilGraph to a plain R function

Lowers a supported subset of `AnvilGraph` objects to a plain R function
(no compilation) suitable for
[`quickr::quick()`](https://rdrr.io/pkg/quickr/man/quick.html). The
returned function expects plain R scalars/vectors/arrays and returns
plain R values/arrays.

## Usage

``` r
graph_to_r_function(graph)
```

## Arguments

- graph:

  ([`AnvilGraph`](https://r-xla.github.io/anvil/dev/reference/AnvilGraph.md))  
  Graph to convert.

## Value

(`function`)

## Details

Most users will prefer
[`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md) with
`backend = "quickr"`. This function is the lower-level graph API.

## See also

[`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md) with
`backend = "quickr"` for tracing and compiling a regular R function in
one step.
