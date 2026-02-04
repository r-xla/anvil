# Primitive Ceiling

Element-wise ceiling. Is the same as
[`nv_ceil()`](https://r-xla.github.io/anvil/reference/nv_ceil.md). You
can also use [`ceiling()`](https://rdrr.io/r/base/Round.html).

## Usage

``` r
nvl_ceil(operand)
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
[`stablehlo::hlo_ceil()`](https://r-xla.github.io/stablehlo/reference/hlo_ceil.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1.2, 2.7, -1.5))
  nvl_ceil(x)
})
#> AnvilTensor
#>   2
#>   3
#>  -1
#> [ CPUf32{3} ] 
```
