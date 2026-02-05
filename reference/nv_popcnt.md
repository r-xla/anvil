# Population Count

Element-wise population count (number of set bits).

## Usage

``` r
nv_popcnt(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_popcnt()`](https://r-xla.github.io/anvil/reference/nvl_popcnt.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(7L, 3L, 15L))
  nv_popcnt(x)
})
#> AnvilTensor
#>  3
#>  2
#>  4
#> [ CPUi32{3} ] 
```
