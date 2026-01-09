# Round

Element-wise rounding.

## Usage

``` r
nv_round(operand, method = "nearest_even")
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- method:

  (`character(1)`)  
  Method to use for rounding. Either `"nearest_even"` (default) or
  `"afz"` (away from zero).

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
