# Sequence

Creates a 1-D array with values from `start` to `end` (inclusive).

Without `steps`, behaves like R's `seq(start, end)` producing integer
values. With `steps`, produces `steps` evenly spaced values (like
`seq(start, end, length.out = steps)`).

## Usage

``` r
nv_seq(start, end, steps = NULL, dtype = NULL, ambiguous = FALSE)
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
  `steps` is given.

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the
  [`vignette("type-promotion")`](https://r-xla.github.io/anvil/dev/articles/type-promotion.md)
  for more details.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
1-D array of length `end - start + 1`.

## Examples

``` r
jit_eval(nv_seq(3, 7))
#> AnvilArray
#>  3
#>  4
#>  5
#>  6
#>  7
#> [ CPUi32{5} ] 
```
