# Primitive Reciprocal Square Root

Element-wise reciprocal square root.

## Usage

``` r
nvl_rsqrt(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish value of data type floating-point.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the input. It is ambiguous if the
input is ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_rsqrt()`](https://r-xla.github.io/stablehlo/reference/hlo_rsqrt.html).

## See also

[`nv_rsqrt()`](https://r-xla.github.io/anvil/dev/reference/nv_rsqrt.md)

## Examples

``` r
jit_eval({
  x <- nv_array(c(1, 4, 9))
  nvl_rsqrt(x)
})
#> AnvilArray
#>  1.0000
#>  0.5000
#>  0.3333
#> [ CPUf32{3} ] 
```
