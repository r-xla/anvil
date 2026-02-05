# Square Root

Element-wise square root. You can also use
[`sqrt()`](https://rdrr.io/r/base/MathFun.html).

## Usage

``` r
nv_sqrt(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_sqrt()`](https://r-xla.github.io/anvil/reference/nvl_sqrt.md) for
the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 4, 9))
  sqrt(x)
})
#> AnvilTensor
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
