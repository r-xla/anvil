# Logistic (Sigmoid)

Element-wise logistic sigmoid: `1 / (1 + exp(-x))`.

## Usage

``` r
nv_logistic(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input.

## See also

[`nvl_logistic()`](https://r-xla.github.io/anvil/reference/nvl_logistic.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(-2, 0, 2))
  nv_logistic(x)
})
#> AnvilTensor
#>  0.1192
#>  0.5000
#>  0.8808
#> [ CPUf32{3} ] 
```
