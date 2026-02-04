# Primitive Log Plus One

Element-wise log(1 + x), more accurate for small x. Is the same as
[`nv_log1p()`](https://r-xla.github.io/anvil/reference/nv_log1p.md).

## Usage

``` r
nvl_log1p(operand)
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
[`stablehlo::hlo_log_plus_one()`](https://r-xla.github.io/stablehlo/reference/hlo_log_plus_one.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(0, 0.001, 1))
  nvl_log1p(x)
})
#> AnvilTensor
#>  0.0000
#>  0.0010
#>  0.6931
#> [ CPUf32{3} ] 
```
