# Print Tensor

Prints a tensor value to the console during JIT execution and returns
the input unchanged. Useful for debugging.

## Usage

``` r
nv_print(operand)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Returns `operand` unchanged.

## Note

Currently only works on the CPU backend.

## See also

[`nvl_print()`](https://r-xla.github.io/anvil/reference/nvl_print.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 2, 3), device = "cpu")
  nv_print(x)
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
