# Primitive Bitcast Convert

Reinterprets the bits of a tensor as a different data type without
modifying the underlying data.

## Usage

``` r
nvl_bitcast_convert(operand, dtype)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of any data type.

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Target data type. If it has the same bit width as the input, the
  output shape is unchanged. If narrower, an extra trailing dimension is
  added. If wider, the last dimension is consumed.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the given `dtype`.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_bitcast_convert()`](https://r-xla.github.io/stablehlo/reference/hlo_bitcast_convert.html).

## See also

[`nv_bitcast_convert()`](https://r-xla.github.io/anvil/reference/nv_bitcast_convert.md)

## Examples

``` r
jit_eval({
  x <- nv_tensor(1L)
  nvl_bitcast_convert(x, dtype = "i8")
})
#> AnvilTensor
#>  1 0 0 0
#> [ CPUi8{1,4} ] 
jit_eval({
  x <- nv_tensor(rep(1L, 4), dtype = "i8")
  nvl_bitcast_convert(x, dtype = "i32")
})
#> AnvilTensor
#>  1.6843e+07
#> [ CPUi32{} ] 
```
