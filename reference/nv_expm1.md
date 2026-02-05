# Exponential Minus One

Element-wise `exp(x) - 1`, more accurate for small `x`.

## Usage

``` r
nv_expm1(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_expm1()`](https://r-xla.github.io/anvil/reference/nvl_expm1.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(0, 0.001, 1))
  nv_expm1(x)
})
#> AnvilTensor
#>  0.0000
#>  0.0010
#>  1.7183
#> [ CPUf32{3} ] 
```
