# Sequence

Creates a 1-D array with values from `start` to `end` (inclusive).

Without `steps`, behaves like R's `seq(start, end)` producing integer
values. With `steps`, produces `steps` evenly spaced values (like
`seq(start, end, length.out = steps)`).

`nv_seq_like()` is a variant where `dtype`, `ambiguous`, and `device`
default to those of `like`.

## Usage

``` r
nv_seq(
  start,
  end,
  steps = NULL,
  dtype = NULL,
  ambiguous = FALSE,
  device = NULL
)

nv_seq_like(
  like,
  start,
  end,
  steps = NULL,
  dtype = NULL,
  ambiguous = NULL,
  device = NULL
)
```

## Arguments

- start, end:

  (`numeric(1)`)  
  Start and end values. When `steps` is `NULL`, must satisfy
  `start <= end`.

- steps:

  (`integer(1)` or `NULL`)  
  Number of evenly spaced values to generate. Must be at least 1. When
  `NULL` (default), generates consecutive integer values from `start` to
  `end`.

- dtype:

  (`character(1)`)  
  Data type. Default `"i32"` when `steps` is `NULL`, `"f32"` when
  `steps` is given. For `nv_seq_like()`, `NULL` uses `dtype(like)`.

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the
  [`vignette("type-promotion")`](https://r-xla.github.io/anvl/dev/articles/type-promotion.md)
  for more details.

- device:

  ( `character(1)` \| `PJRTDevice` \|
  [`quickr_device`](https://r-xla.github.io/anvl/dev/reference/quickr_device.md)
  \| `NULL`)  
  Device for data to live on.

- like:

  ([`AnvlArray`](https://r-xla.github.io/anvl/dev/reference/AnvlArray.md))  
  Existing array whose attributes are used as defaults (only for
  `nv_seq_like()`).

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
1-D array of length `end - start + 1`.

## Examples

``` r
nv_seq(3, 7)
#> AnvlArray
#>  3
#>  4
#>  5
#>  6
#>  7
#> [ CPUi32{5} ] 
x <- nv_array(c(1, 2, 3), dtype = "f64")
nv_seq_like(x, 1, 5)
#> AnvlArray
#>  1
#>  2
#>  3
#>  4
#>  5
#> [ CPUf64{5} ] 
```
