# HloEnv

Environment for storing graph value to func value mappings. This is a
mutable class.

## Usage

``` r
HloEnv(parent = NULL, gval_to_fval = NULL)
```

## Arguments

- parent:

  (`HloEnv` \| `NULL`)  
  Parent environment for lookups.

- gval_to_fval:

  (`hashtab`)  
  Mapping from graph values to func values.

## Value

(`HloEnv`)
