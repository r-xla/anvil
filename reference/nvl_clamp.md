# Primitive Clamp

Element-wise clamp: max(min_val, min(operand, max_val)).

## Usage

``` r
nvl_clamp(min_val, operand, max_val)
```

## Arguments

- min_val:

  ([`tensorish`](tensorish.md))  
  Minimum value (scalar or same shape as operand).

- operand:

  ([`tensorish`](tensorish.md))  
  Operand.

- max_val:

  ([`tensorish`](tensorish.md))  
  Maximum value (scalar or same shape as operand).

## Value

[`tensorish`](tensorish.md)
