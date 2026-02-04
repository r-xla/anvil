# Primitive Clamp

Element-wise clamp: max(min_val, min(operand, max_val)).

## Usage

``` r
nvl_clamp(min_val, operand, max_val)
```

## Arguments

- min_val:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Minimum value (scalar or same shape as operand).

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- max_val:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Maximum value (scalar or same shape as operand).

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)

## Shapes

`min_val` and `max_val` must each be either scalar or the same shape as
`operand`. The output has the same shape as `operand`.

## StableHLO

Calls
[`stablehlo::hlo_clamp()`](https://r-xla.github.io/stablehlo/reference/hlo_clamp.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(-1, 0.5, 2))
  nvl_clamp(nv_scalar(0), x, nv_scalar(1))
})
#> AnvilTensor
#>  0.0000
#>  0.5000
#>  1.0000
#> [ CPUf32{3} ] 
```
