# Median

Computes the median along a dimension. Equivalent to
`nv_quantile(x, 0.5, dim, interpolation)`; for an even-length axis with
the default `"linear"` interpolation, the average of the two middle
values is returned, matching base R's
[`median()`](https://rdrr.io/r/stats/median.html).

You can also use [`median()`](https://rdrr.io/r/stats/median.html)
directly on an
[`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md)
or [`AnvlBox`](https://r-xla.github.io/anvl/dev/reference/AnvlBox.md);
extra arguments (e.g. `interpolation`) are forwarded via `...`.

## Usage

``` r
nv_median(x, dim = NULL, interpolation = "linear")

# S3 method for class 'AnvlBox'
median(x, na.rm = FALSE, ...)

# S3 method for class 'AnvlArray'
median(x, na.rm = FALSE, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  The array.

- dim:

  (`integer(1)` \| `NULL`)  
  Dimension along which to compute the median. If `NULL` (default), uses
  the last dimension.

- interpolation:

  (`character(1)`)  
  Forwarded to
  [`nv_quantile()`](https://r-xla.github.io/anvl/dev/reference/nv_quantile.md).
  One of `"linear"` (default), `"lower"`, `"higher"`, `"nearest"`,
  `"midpoint"`.

- na.rm:

  Included for compatibility with the
  [`stats::median()`](https://rdrr.io/r/stats/median.html) generic. anvl
  arrays do not carry `NA`s; passing `na.rm = TRUE` raises an error.

- ...:

  Forwarded to `nv_median()`.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Same shape as `x` with `dim` removed.

## See also

[`nv_quantile()`](https://r-xla.github.io/anvl/dev/reference/nv_quantile.md),
[`nv_sort()`](https://r-xla.github.io/anvl/dev/reference/nv_sort.md),
[`prim_sort()`](https://r-xla.github.io/anvl/dev/reference/prim_sort.md).

## Examples

``` r
nv_median(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#> AnvlArray
#>  3.5000
#> [ CPUf32{} ] 
median(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#> AnvlArray
#>  3.5000
#> [ CPUf32{} ] 
nv_median(nv_matrix(c(3, 1, 5, 2, 4, 0), nrow = 2, byrow = TRUE),
  dim = 2L
)
#> AnvlArray
#>  3
#>  2
#> [ CPUf32{2} ] 
# forwards through the S3 generic via `...`
median(nv_array(c(1, 2, 3, 4)), interpolation = "lower")
#> AnvlArray
#>  2
#> [ CPUf32{} ] 
```
