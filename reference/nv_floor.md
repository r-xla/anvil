# Floor

Element-wise floor (round toward negative infinity). You can also use
[`floor()`](https://rdrr.io/r/base/Round.html).

## Usage

``` r
nv_floor(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_floor()`](https://r-xla.github.io/anvil/reference/nvl_floor.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1.2, 2.7, -1.5))
  floor(x)
})
#> AnvilTensor
#>   1
#>   2
#>  -2
#> [ CPUf32{3} ] 
```
