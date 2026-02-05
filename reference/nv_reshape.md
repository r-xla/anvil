# Reshape

Reshapes a tensor to a new shape without changing the underlying data.
Returns the input unchanged if it already has the target shape.

## Usage

``` r
nv_reshape(operand, shape)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Target shape. Must have the same number of elements as `operand`.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the given `shape` and the same data type as `operand`.

## Details

Note that row-major order is used, which differs from R's column-major
order.

## See also

[`nvl_reshape()`](https://r-xla.github.io/anvil/reference/nvl_reshape.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(1:6)
  nv_reshape(x, c(2, 3))
})
#> AnvilTensor
#>  1 2 3
#>  4 5 6
#> [ CPUi32{2,3} ] 
```
