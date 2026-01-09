# Primitive If

Conditional execution of branches.

## Usage

``` r
nvl_if(pred, true, false)
```

## Arguments

- pred:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Scalar boolean predicate.

- true:

  (`expression`)  
  Expression for true branch.

- false:

  (`expression`)  
  Expression for false branch.

## Value

Result of the executed branch.
