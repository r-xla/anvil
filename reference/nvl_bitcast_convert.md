# Primitive Bitcast Convert

Reinterprets tensor bits as a different dtype.

## Usage

``` r
nvl_bitcast_convert(operand, dtype)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)

## Shapes

If the source and target types have the same bit width, the output has
the same shape as `operand`. Otherwise the last dimension is adjusted
based on the bit-width ratio.

## StableHLO

Calls
[`stablehlo::hlo_bitcast_convert()`](https://r-xla.github.io/stablehlo/reference/hlo_bitcast_convert.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(1L)
  nvl_bitcast_convert(x, dtype = "f32")
})
#> AnvilTensor
#>  1.4013e-45
#> [ CPUf32{1} ] 
```
