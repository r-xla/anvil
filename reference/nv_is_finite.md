# Is Finite

Element-wise check if values are finite (not `Inf`, `-Inf`, or `NaN`).

## Usage

``` r
nv_is_finite(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape as the input and boolean data type.

## See also

[`nvl_is_finite()`](https://r-xla.github.io/anvil/reference/nvl_is_finite.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, Inf, NaN, -Inf, 0))
  nv_is_finite(x)
})
#> AnvilTensor
#>  1
#>  0
#>  0
#>  0
#>  1
#> [ CPUi1{5} ] 
```
