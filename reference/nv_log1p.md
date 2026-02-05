# Log Plus One

Element-wise `log(1 + x)`, more accurate for small `x`.

## Usage

``` r
nv_log1p(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_log1p()`](https://r-xla.github.io/anvil/reference/nvl_log1p.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(0, 0.001, 1))
  nv_log1p(x)
})
#> AnvilTensor
#>  0.0000
#>  0.0010
#>  0.6931
#> [ CPUf32{3} ] 
```
