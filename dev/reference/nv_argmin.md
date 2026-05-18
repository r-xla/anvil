# Index of the Minimum

Returns the index of the minimum value along a dimension. Ties are
broken by returning the smallest index.

## Usage

``` r
nv_argmin(operand, dim = NULL, drop = TRUE)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Operand.

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
Same shape as `operand` with `dim` removed (or set to 1 if
`drop = FALSE`).

## See also

[`nv_argmax()`](https://r-xla.github.io/anvl/dev/reference/nv_argmax.md),
[`nv_reduce_min()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_min.md).

## Examples

``` r
nv_argmin(nv_array(c(3, 1, 4, 1, 5, 9, 2, 6)))
#> AnvlArray
#>  2
#> [ CPUi32{} ] 
```
