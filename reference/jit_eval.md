# JIT-compile and evaluate an expression

Convenience wrapper that JIT-compiles and immediately evaluates a single
expression. Equivalent to wrapping `expr` in an anonymous function,
calling [`jit()`](https://r-xla.github.io/anvil/reference/jit.md) on it,
and invoking the result. Useful if you want to evaluate an expression
once.

## Usage

``` r
jit_eval(expr, device = NULL)
```

## Arguments

- expr:

  (NSE)  
  Expression to compile and evaluate.

- device:

  (`NULL` \| `character(1)` \|
  [`PJRTDevice`](https://r-xla.github.io/pjrt/reference/pjrt_device.html))  
  The device to use. By default (`NULL`), the device is inferred from
  the tensors encountered during tracing, falling back to `"cpu"`. or
  `"cpu"`.

## Value

(`any`)  
Result of the compiled and evaluated expression.

## Examples

``` r
x <- nv_tensor(c(1, 2, 3), dtype = "f32")
jit_eval(x + x)
#> AnvilTensor
#>  2
#>  4
#>  6
#> [ CPUf32{3} ] 
```
