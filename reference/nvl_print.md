# Primitive Print

Prints a tensor during execution. Returns the input unchanged. Note:
Currently only works on CPU backend.

## Usage

``` r
nvl_print(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)

## Shapes

Output has the same shape as `operand`.

## StableHLO

Uses
[`stablehlo::hlo_custom_call()`](https://r-xla.github.io/stablehlo/reference/hlo_custom_call.html)
internally.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 2, 3), device = "cpu")
  nvl_print(x)
})
#> AnvilTensor
#>  1
#>  2
#>  3
#> [ f32{3} ]
#> AnvilTensor
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
