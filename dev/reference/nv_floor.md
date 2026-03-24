# Floor

Element-wise floor (round toward negative infinity). You can also use
[`floor()`](https://rdrr.io/r/base/Round.html).

## Usage

``` r
nv_floor(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_floor()`](https://r-xla.github.io/anvil/dev/reference/nvl_floor.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(c(1.2, 2.7, -1.5))
  floor(x)
})
#> AnvilArray
#>   1
#>   2
#>  -2
#> [ CPUf32{3} ] 
```
