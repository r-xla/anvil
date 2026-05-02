# Index of the Maximum

Returns the index of the maximum value along a dimension. Ties are
broken by returning the smallest index.

## Usage

``` r
nv_argmax(x, dim = NULL, drop = TRUE)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  The array.

- dim:

  (`integer(1)` \| `NULL`)  
  Dimension along which to find the index. If `NULL` (default), uses the
  last dimension.

- drop:

  (`logical(1)`)  
  If `TRUE` (default) the reduced dimension is removed; if `FALSE` it is
  kept with size 1.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md) of
dtype `i32`  
Same shape as `x` with `dim` removed (or set to 1 if `drop = FALSE`).

## See also

[`nv_argmin()`](https://r-xla.github.io/anvl/dev/reference/nv_argmin.md),
[`nv_reduce_max()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_max.md).

## Examples

``` r
nv_argmax(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#> AnvlArray
#>  6
#> [ CPUi32{} ] 
nv_argmax(nv_array(matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE)),
  dim = 2L
)
#> AnvlArray
#>  3
#>  2
#> [ CPUi32{2} ] 
```
