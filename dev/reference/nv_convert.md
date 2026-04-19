# Convert Data Type

Converts the elements of an array to a different data type. Returns the
input unchanged if it already has the target type.

## Usage

``` r
nv_convert(operand, dtype)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

- dtype:

  (`character(1)` \|
  [`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Data type.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the given `dtype` and the same shape as `operand`.

## See also

[`nvl_convert()`](https://r-xla.github.io/anvil/dev/reference/nvl_convert.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(c(1L, 2L, 3L))
  nv_convert(x, dtype = "f32")
})
#> AnvilArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
