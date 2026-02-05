# Bitcast Conversion

Reinterprets the bits of a tensor as a different data type without
modifying the underlying data. If the target type is narrower, an extra
trailing dimension is added; if wider, the last dimension is consumed.

## Usage

``` r
nv_bitcast_convert(operand, dtype)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Target data type.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the given `dtype`.

## See also

[`nvl_bitcast_convert()`](https://r-xla.github.io/anvil/reference/nvl_bitcast_convert.md)
for the underlying primitive,
[`nv_convert()`](https://r-xla.github.io/anvil/reference/nv_convert.md)
for value-preserving type conversion.

## Examples

``` r
jit_eval({
  x <- nv_tensor(1L)
  nvl_bitcast_convert(x, dtype = "i8")
})
#> AnvilTensor
#>  1 0 0 0
#> [ CPUi8{1,4} ] 
```
