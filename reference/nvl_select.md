# Primitive Select

Selects elements based on a predicate.

## Usage

``` r
nvl_select(pred, true_value, false_value)
```

## Arguments

- pred:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Boolean predicate tensor.

- true_value:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Value when pred is true.

- false_value:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Value when pred is false.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
