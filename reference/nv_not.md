# Logical Not

Element-wise logical NOT. You can also use the `!` operator.

## Usage

``` r
nv_not(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_not()`](https://r-xla.github.io/anvil/reference/nvl_not.md) for
the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(TRUE, FALSE, TRUE))
  !x
})
#> AnvilTensor
#>  0
#>  1
#>  0
#> [ CPUi1{3} ] 
```
