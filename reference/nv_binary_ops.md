# Binary Operations

Binary operations on tensors.

## Usage

``` r
nv_add(lhs, rhs)

nv_mul(lhs, rhs)

nv_sub(lhs, rhs)

nv_div(lhs, rhs)

nv_pow(lhs, rhs)

nv_eq(lhs, rhs)

nv_ne(lhs, rhs)

nv_gt(lhs, rhs)

nv_ge(lhs, rhs)

nv_lt(lhs, rhs)

nv_le(lhs, rhs)

nv_max(lhs, rhs)

nv_min(lhs, rhs)

nv_remainder(lhs, rhs)

nv_and(lhs, rhs)

nv_or(lhs, rhs)

nv_xor(lhs, rhs)

nv_shift_left(lhs, rhs)

nv_shift_right_logical(lhs, rhs)

nv_shift_right_arithmetic(lhs, rhs)

nv_atan2(lhs, rhs)
```

## Arguments

- lhs:

  ([`nv_tensor`](nv_tensor.md))

- rhs:

  ([`nv_tensor`](nv_tensor.md))

## Value

[`nv_tensor`](nv_tensor.md)

## Examples

``` r
# Comparison operators such `nv_eq`, `nv_le`, `nv_gt`, etc
# are nondifferentiable and contribute zero to gradients.
relu <- function(x) {
  nv_convert(x > nv_scalar(0), "f32")*x
}
# df/dx = 1 if x > 0 else 0
g_relu <- jit(gradient(relu, "x"))

g_relu(nv_scalar(1, dtype = "f32"))
#> $x
#> AnvilTensor 
#>  1.0000
#> [ CPUf32{} ] 
#> 
g_relu(nv_scalar(-1, dtype = "f32"))
#> $x
#> AnvilTensor 
#>  0.0000
#> [ CPUf32{} ] 
#> 
```
