# While

Functional while loop.

## Usage

``` r
nv_while(init, cond, body)
```

## Arguments

- init:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  Initial state.

- cond:

  (`function`)  
  Condition function: `f: state -> bool`.

- body:

  (`function`)  
  Body function. `f: state -> state`.

## Value

[`tensorish`](tensorish.md)
