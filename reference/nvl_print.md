# Primitive Print

Prints a tensor value to the console during execution and returns the
input unchanged. This is useful for debugging JIT-compiled code.

## Usage

``` r
nvl_print(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of any data type.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Returns `operand` as-is.

## Note

Currently only works on the CPU backend.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to
[`stablehlo::hlo_custom_call()`](https://r-xla.github.io/stablehlo/reference/hlo_custom_call.html).

## See also

[`nv_print()`](https://r-xla.github.io/anvil/reference/nv_print.md)

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
