# Trace, lower, and compile a function to an XLA executable

Takes a function, traces it into a computational graph, lowers it to
StableHLO, and compiles it to a PJRT executable. Returns the compiled
executable along with metadata needed for execution.

## Usage

``` r
compile_to_xla(f, args_flat, in_tree, donate = character(), device = NULL)
```

## Arguments

- f:

  (`function`)  
  Function to compile.

- args_flat:

  (`list`)  
  Flat list of abstract input values.

- in_tree:

  (`Node`)  
  Tree structure of the inputs.

- donate:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of the arguments whose buffers should be donated.

- device:

  (`NULL` \| `character(1)`)  
  Target device (e.g. `"cpu"`, `"cuda"`). If `NULL`, inferred from
  traced tensors.

## Value

A `list` with elements:

- `exec`: The compiled PJRT executable.

- `out_tree`: The output tree structure.

- `const_tensors`: Constants needed at execution time.

- `ambiguous_out`: Logical vector indicating which outputs are ambiguous
  (`NULL` if none are).
