# Unary Operations

Unary operations on tensors.

## Usage

``` r
nv_neg(operand)

nv_abs(operand)

nv_sqrt(operand)

nv_rsqrt(operand)

nv_log(operand)

nv_tanh(operand)

nv_tan(operand)

nv_sine(operand)

nv_cosine(operand)

nv_floor(operand)

nv_ceil(operand)

nv_sign(operand)

nv_exp(operand)

nv_round(operand, method = "nearest_even")
```

## Arguments

- operand:

  ([`nv_tensor`](nv_tensor.md))  
  Operand.

- method:

  (`character(1)`)  
  Method to use for rounding. Either `"nearest_even"` (default) or
  `"afz"` (away from zero).

## Value

[`nv_tensor`](nv_tensor.md)
