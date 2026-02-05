# Primitive Clamp

Clamps every element of `operand` to the range `[min_val, max_val]`,
i.e. `max(min_val, min(operand, max_val))`.

## Usage

``` r
nvl_clamp(min_val, operand, max_val)
```

## Arguments

- min_val:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Minimum value. Must be scalar or the same shape as `operand`.

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of any data type.

- max_val:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Maximum value. Must be scalar or the same shape as `operand`.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same data type and shape as `operand`. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_clamp()`](https://r-xla.github.io/stablehlo/reference/hlo_clamp.html).

## See also

[`nv_clamp()`](https://r-xla.github.io/anvil/reference/nv_clamp.md)

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
