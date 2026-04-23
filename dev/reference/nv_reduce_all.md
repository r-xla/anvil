# All Reduction

Performs logical AND along the specified dimensions. Returns `TRUE` only
if all elements are `TRUE`.

## Usage

``` r
nv_reduce_all(operand, dims, drop = TRUE)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Operand.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions to reduce.

- drop:

  (`logical(1)`)  
  Whether to drop reduced dimensions.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Boolean array. When `drop = TRUE`, the reduced dimensions are removed.
When `drop = FALSE`, the reduced dimensions are set to 1.

## See also

[`prim_reduce_all()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce_all.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(matrix(c(TRUE, FALSE, TRUE, TRUE), nrow = 2))
nv_reduce_all(x, dims = 1L)
#> AnvlArray
#>  0
#>  1
#> [ CPUbool{2} ] 
```
