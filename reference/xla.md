# Compile a function to an XLA executable and wrap it as an R function

Takes a function, traces it, translates to StableHLO, compiles to an XLA
executable, and returns an R function that executes it.

## Usage

``` r
xla(f, args, donate = character(), device = NULL)
```

## Arguments

- f:

  (`function`)  
  Function to compile.

- args:

  (`list`)  
  List of abstract input values (as passed to `f`).

- donate:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Names of the arguments whose buffers should be donated.

- device:

  (`character(1)`)  
  Target device such as `"cpu"` (default) or `"cuda"`.

## Value

(`function`)  
A function that accepts
[`AnvilTensor`](https://r-xla.github.io/anvil/reference/AnvilTensor.md)
arguments (matching the flat inputs) and returns the result as
[`AnvilTensor`](https://r-xla.github.io/anvil/reference/AnvilTensor.md)s.
