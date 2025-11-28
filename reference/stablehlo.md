# Lower a function to StableHLO

Immediately lower a flattened function to a StableHLO Func object.

## Usage

``` r
stablehlo(graph, constants_as_inputs = TRUE, env = NULL, donate = character())
```

## Arguments

- graph:

  (`Graph`)  
  The graph to lower.

- constants_as_inputs:

  (`logical(1)`)  
  Whether to add constants as inputs.

- env:

  (`HloEnv` \| `NULL`)  
  The environment for storing graph value to func variable mappings.

- donate:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of the arguments whose buffers should be donated. Donated
  buffers can be aliased with outputs of the same type.

## Value

(`list`) with elements:

- `func`: The StableHLO `Func` object

- `constants`: The constants of the graph
