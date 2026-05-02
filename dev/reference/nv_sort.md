# Sort

Sorts an array along a dimension.

You can also use [`sort()`](https://rdrr.io/r/base/sort.html) directly.

## Usage

``` r
nv_sort(x, dim = NULL, decreasing = FALSE)

# S3 method for class 'AnvlBox'
sort(x, decreasing = FALSE, ...)

# S3 method for class 'AnvlArray'
sort(x, decreasing = FALSE, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  The array to sort.

- dim:

  (`integer(1)` \| `NULL`)  
  Dimension along which to sort. If `NULL` (default), uses the last
  dimension.

- decreasing:

  (`logical(1)`)  
  If `TRUE`, sort in decreasing order. Default `FALSE`.

- ...:

  Forwarded to `nv_sort()`.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Same shape and data type as `x`.

## See also

[`prim_sort()`](https://r-xla.github.io/anvl/dev/reference/prim_sort.md)
for the underlying primitive,
[`nv_argsort()`](https://r-xla.github.io/anvl/dev/reference/nv_argsort.md)
(where sort stability is observable),
[`nv_top_k()`](https://r-xla.github.io/anvl/dev/reference/nv_top_k.md),
[`nv_median()`](https://r-xla.github.io/anvl/dev/reference/nv_median.md),
[`nv_argmax()`](https://r-xla.github.io/anvl/dev/reference/nv_argmax.md),
[`nv_argmin()`](https://r-xla.github.io/anvl/dev/reference/nv_argmin.md).

## Examples

``` r
x <- nv_array(c(3, 1, 4, 1, 5, 9, 2, 6))
nv_sort(x)
#> AnvlArray
#>  1
#>  1
#>  2
#>  3
#>  4
#>  5
#>  6
#>  9
#> [ CPUf32{8} ] 
sort(x) # via the S3 generic
#> AnvlArray
#>  1
#>  1
#>  2
#>  3
#>  4
#>  5
#>  6
#>  9
#> [ CPUf32{8} ] 
nv_sort(x, decreasing = TRUE)
#> AnvlArray
#>  9
#>  6
#>  5
#>  4
#>  3
#>  2
#>  1
#>  1
#> [ CPUf32{8} ] 

m <- nv_array(matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE))
nv_sort(m, dim = 2L)
#> AnvlArray
#>  1 3 5
#>  0 2 4
#> [ CPUf32{2,3} ] 
```
