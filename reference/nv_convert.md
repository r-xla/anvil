# Convert Data Type

Converts the elements of a tensor to a different data type. Returns the
input unchanged if it already has the target type.

## Usage

``` r
nv_convert(operand, dtype)
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
Has the given `dtype` and the same shape as `operand`.

## See also

[`nvl_convert()`](https://r-xla.github.io/anvil/reference/nvl_convert.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1L, 2L, 3L))
  nv_convert(x, dtype = "f32")
})
#> AnvilTensor
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
