# Primitive Not

Element-wise logical NOT.

## Usage

``` r
nvl_not(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of data type boolean, integer, or unsigned integer.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_not()`](https://r-xla.github.io/stablehlo/reference/hlo_not.html).

## See also

[`nv_not()`](https://r-xla.github.io/anvil/reference/nv_not.md)

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(TRUE, FALSE, TRUE))
  nvl_not(x)
})
#> AnvilTensor
#>  0
#>  1
#>  0
#> [ CPUi1{3} ] 
```
