# Primitive Floor

Element-wise floor. Is the same as
[`nv_floor()`](https://r-xla.github.io/anvil/reference/nv_floor.md). You
can also use [`floor()`](https://rdrr.io/r/base/Round.html).

## Usage

``` r
nvl_floor(operand)
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
[`stablehlo::hlo_floor()`](https://r-xla.github.io/stablehlo/reference/hlo_floor.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1.2, 2.7, -1.5))
  nvl_floor(x)
})
#> AnvilTensor
#>   1
#>   2
#>  -2
#> [ CPUf32{3} ] 
```
