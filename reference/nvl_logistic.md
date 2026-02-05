# Primitive Logistic (Sigmoid)

Element-wise logistic sigmoid: 1 / (1 + exp(-x)).

## Usage

``` r
nvl_logistic(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of data type floating-point.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_logistic()`](https://r-xla.github.io/stablehlo/reference/hlo_logistic.html).

## See also

[`nv_logistic()`](https://r-xla.github.io/anvil/reference/nv_logistic.md)

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(-2, 0, 2))
  nvl_logistic(x)
})
#> AnvilTensor
#>  0.1192
#>  0.5000
#>  0.8808
#> [ CPUf32{3} ] 
```
