# Bitcast Conversion

Reinterprets the bits of an array as a different data type without
modifying the underlying data. If the target type is narrower, an extra
trailing dimension is added; if wider, the last dimension is consumed.

## Usage

``` r
nv_bitcast_convert(operand, dtype)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

- dtype:

  (`character(1)` \|
  [`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Target data type.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the given `dtype`.

## See also

[`nvl_bitcast_convert()`](https://r-xla.github.io/anvil/dev/reference/nvl_bitcast_convert.md)
for the underlying primitive,
[`nv_convert()`](https://r-xla.github.io/anvil/dev/reference/nv_convert.md)
for value-preserving type conversion.

## Examples

``` r
jit_eval({
  x <- nv_array(1L)
  nvl_bitcast_convert(x, dtype = "i8")
})
#> AnvilArray
#>  1 0 0 0
#> [ CPUi8{1,4} ] 
```
