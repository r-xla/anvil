# Any Reduction

Performs logical OR along the specified dimensions. Returns `TRUE` if
any element is `TRUE`.

## Usage

``` r
nv_reduce_any(operand, dims, drop = TRUE)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions to reduce.

- drop:

  (`logical(1)`)  
  Whether to drop reduced dimensions.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Boolean tensor. When `drop = TRUE`, the reduced dimensions are removed.
When `drop = FALSE`, the reduced dimensions are set to 1.

## See also

[`nvl_reduce_any()`](https://r-xla.github.io/anvil/reference/nvl_reduce_any.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2))
  nv_reduce_any(x, dims = 1L)
})
#> AnvilTensor
#>  1
#>  1
#> [ CPUi1{2} ] 
```
