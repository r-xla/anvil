# Logical Not

Element-wise logical NOT. You can also use the `!` operator.

## Usage

``` r
nv_not(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_not()`](https://r-xla.github.io/anvil/dev/reference/nvl_not.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(c(TRUE, FALSE, TRUE))
  !x
})
#> AnvilArray
#>  0
#>  1
#>  0
#> [ CPUbool{3} ] 
```
