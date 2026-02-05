# Clamp

Element-wise clamp: max(min_val, min(operand, max_val)). Converts
`min_val` and `max_val` to the data type of `operand`.

## Usage

``` r
nv_clamp(min_val, operand, max_val)
```

## Arguments

- min_val:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Minimum value.

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- max_val:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Maximum value.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)

## Details

The underlying stableHLO function already broadcasts scalars, so no need
to broadcast manually.
