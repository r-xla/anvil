# Print Array

Prints an array value to the console during JIT execution and returns
the input unchanged. Useful for debugging.

## Usage

``` r
nv_print(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Returns `operand` unchanged.

## See also

[`nvl_print()`](https://r-xla.github.io/anvil/dev/reference/nvl_print.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(c(1, 2, 3))
  nv_print(x)
})
#> AnvilArray
#>  1
#>  2
#>  3
#> [ f32{3} ]
#> AnvilArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
