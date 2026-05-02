# Min Reduction

Finds the minimum of array elements along the specified dimensions.

## Usage

``` r
nv_reduce_min(operand, dims = NULL, drop = TRUE)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Operand.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Dimensions to reduce. If `NULL` (default), reduces over all
  dimensions, returning a scalar.

- drop:

  (`logical(1)`)  
  Whether to drop reduced dimensions.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type as the input. When `drop = TRUE`, the reduced
dimensions are removed. When `drop = FALSE`, the reduced dimensions are
set to 1.

## See also

[`prim_reduce_min()`](https://r-xla.github.io/anvl/dev/reference/prim_reduce_min.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(matrix(1:6, nrow = 2))
nv_reduce_min(x)            # all dims -> scalar
#> AnvlArray
#>  1
#> [ CPUi32{} ] 
nv_reduce_min(x, dims = 1L)
#> AnvlArray
#>  1
#>  3
#>  5
#> [ CPUi32{3} ] 
```
