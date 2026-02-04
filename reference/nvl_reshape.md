# Primitive Reshape

Reshapes a tensor to a new shape.

## Usage

``` r
nvl_reshape(operand, shape)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)

## Shapes

`shape` must have the same number of elements as `operand`.

## StableHLO

Calls
[`stablehlo::hlo_reshape()`](https://r-xla.github.io/stablehlo/reference/hlo_reshape.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(1:6)
  nvl_reshape(x, shape = c(2, 3))
})
#> AnvilTensor
#>  1 2 3
#>  4 5 6
#> [ CPUi32{2,3} ] 
```
