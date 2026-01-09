# Clamp

Element-wise clamp: max(min_val, min(operand, max_val)).

## Usage

``` r
nv_clamp(min_val, operand, max_val)
```

## Arguments

- min_val:

  ([`tensorish`](tensorish.md))  
  Minimum value.

- operand:

  ([`tensorish`](tensorish.md))  
  Operand.

- max_val:

  ([`tensorish`](tensorish.md))  
  Maximum value.

## Value

[`tensorish`](tensorish.md)

## Details

The underlying stableHLO function already broadcasts scalars, so no need
to broadcast manually.
