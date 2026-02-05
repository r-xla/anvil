# Round

Element-wise rounding. You can also use the
[`round()`](https://rdrr.io/r/base/Round.html) generic.

## Usage

``` r
nv_round(operand, method = "nearest_even")
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- method:

  (`character(1)`)  
  Rounding method. Either `"nearest_even"` (default) or `"afz"` (away
  from zero).

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_round()`](https://r-xla.github.io/anvil/reference/nvl_round.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1.4, 2.5, 3.6))
  round(x)
})
#> AnvilTensor
#>  1
#>  2
#>  4
#> [ CPUf32{3} ] 
```
