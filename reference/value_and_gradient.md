# Value and Gradient

Returns a new function that computes both the output of `f` and its
gradient in a single forward+backward pass. The result is a named list
with elements `value` (the original return value of `f`) and `grad` (the
gradients, structured like the inputs or the `wrt` subset).

## Usage

``` r
value_and_gradient(f, wrt = NULL)
```

## Arguments

- f:

  (`function`)  
  Function to differentiate. Arguments can be tensorish
  ([`AnvilTensor`](https://r-xla.github.io/anvil/reference/AnvilTensor.md))
  or static (non-tensor) values. Must return a single scalar float
  tensor.

- wrt:

  (`character` or `NULL`)  
  Names of the arguments to compute the gradient with respect to. Only
  tensorish (float tensor) arguments can be included; static arguments
  must not appear in `wrt`. If `NULL` (the default), the gradient is
  computed with respect to all arguments (which must all be tensorish in
  that case).

## Value

A function with the same formals as `f` that returns
`list(value = ..., grad = ...)`.

## See also

[`gradient()`](https://r-xla.github.io/anvil/reference/gradient.md)

## Examples

``` r
loss_fn <- function(x) sum(x^2L)
vg <- jit(value_and_gradient(loss_fn))
result <- vg(nv_tensor(c(3, 4), dtype = "f32"))
result$value
#> AnvilTensor
#>  25
#> [ CPUf32{} ] 
result$grad
#> $x
#> AnvilTensor
#>  6
#>  8
#> [ CPUf32{2} ] 
#> 
```
