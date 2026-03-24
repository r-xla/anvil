# Primitive Multiplication

Multiplies two arrays element-wise.

## Usage

``` r
nvl_mul(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish values of any data type. Must have the same shape.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same shape and data type as the inputs. It is ambiguous if both
inputs are ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_multiply()`](https://r-xla.github.io/stablehlo/reference/hlo_multiply.html).

## See also

[`nv_mul()`](https://r-xla.github.io/anvil/dev/reference/nv_mul.md), `*`

## Examples

``` r
jit_eval({
  x <- nv_array(c(1, 2, 3))
  y <- nv_array(c(4, 5, 6))
  nvl_mul(x, y)
})
#> AnvilArray
#>   4
#>  10
#>  18
#> [ CPUf32{3} ] 
```
