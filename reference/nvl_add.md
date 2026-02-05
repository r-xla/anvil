# Primitive Addition

Adds two tensors element-wise.

## Usage

``` r
nvl_add(lhs, rhs)
```

## Arguments

- lhs, rhs:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish values of any data type. Must have the same shape.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same shape and data type as the inputs. It is ambiguous if both
inputs are ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_add()`](https://r-xla.github.io/stablehlo/reference/hlo_add.html).

## See also

[`nv_add()`](https://r-xla.github.io/anvil/reference/nv_add.md), `+`

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 2, 3))
  y <- nv_tensor(c(4, 5, 6))
  nvl_add(x, y)
})
#> AnvilTensor
#>  5
#>  7
#>  9
#> [ CPUf32{3} ] 
```
