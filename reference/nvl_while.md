# Primitive While Loop

Executes a while loop.

## Usage

``` r
nvl_while(init, cond, body)
```

## Arguments

- init:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  Named list of initial state values.

- cond:

  (`function`)  
  Condition function returning boolean.

- body:

  (`function`)  
  Body function returning updated state.

## Value

Final state after loop terminates.
