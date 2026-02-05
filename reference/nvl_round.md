# Primitive Round

Rounds the elements of a tensor to the nearest integer.

## Usage

``` r
nvl_round(operand, method = "nearest_even")
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of data type floating-point.

- method:

  (`character(1)`)  
  Rounding method. `"nearest_even"` (default) rounds to the nearest even
  integer on a tie, `"afz"` rounds away from zero on a tie.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same dtype and shape as `operand`. It is ambiguous if the input
is ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_round_nearest_even()`](https://r-xla.github.io/stablehlo/reference/hlo_round_nearest_even.html)
or
[`stablehlo::hlo_round_nearest_afz()`](https://r-xla.github.io/stablehlo/reference/hlo_round_nearest_afz.html)
depending on the `method` parameter.

## See also

[`nv_round()`](https://r-xla.github.io/anvil/reference/nv_round.md)

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1.4, 2.5, 3.6))
  nvl_round(x)
})
#> AnvilTensor
#>  1
#>  2
#>  4
#> [ CPUf32{3} ] 
```
