# Convert an AnvilGraph to a quickr-compiled function

Lowers a supported subset of `AnvilGraph` objects to a plain R function
and compiles it with
[`quickr::quick()`](https://rdrr.io/pkg/quickr/man/quick.html).

## Usage

``` r
graph_to_quickr_function(graph)
```

## Arguments

- graph:

  ([`AnvilGraph`](https://r-xla.github.io/anvil/dev/reference/AnvilGraph.md))  
  Graph to convert.

## Value

(`function`)

## Details

The returned function expects plain R scalars/vectors/arrays (not
[`AnvilTensor`](https://r-xla.github.io/anvil/dev/reference/AnvilTensor.md))
and returns plain R values/arrays.

If the graph returns multiple outputs (e.g. a nested list), the compiled
function returns the same structure by rebuilding the output tree in R.

For a list of supported primitives see
[`vignette("primitives")`](https://r-xla.github.io/anvil/dev/articles/primitives.md).

Supported dtypes are `f64`, `i32`, and `pred`. The code generator
currently supports tensors up to rank 5. Some primitives are more
restricted (e.g. `transpose` currently only handles rank-2 tensors).

Most users will prefer
[`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md) with
`backend = "quickr"`. This function is the lower-level graph API.

## See also

[`jit()`](https://r-xla.github.io/anvil/dev/reference/jit.md) with
`backend = "quickr"` for tracing and compiling a regular R function in
one step.
