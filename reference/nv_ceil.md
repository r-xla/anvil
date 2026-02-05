# Ceiling

Element-wise ceiling (round toward positive infinity). You can also use
[`ceiling()`](https://rdrr.io/r/base/Round.html).

## Usage

``` r
nv_ceil(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_ceil()`](https://r-xla.github.io/anvil/reference/nvl_ceil.md) for
the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1.2, 2.7, -1.5))
  ceiling(x)
})
#> AnvilTensor
#>   2
#>   3
#>  -1
#> [ CPUf32{3} ] 
```
