# Sequence

Creates a 1-D array with integer values from `start` to `end`
(inclusive), analogous to R's `seq(start, end)`.

## Usage

``` r
nv_seq(start, end, dtype = "i32", ambiguous = FALSE)
```

## Arguments

- start, end:

  (`integer(1)`)  
  Start and end values. Must satisfy `start <= end`.

- dtype:

  (`character(1)` \|
  [`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Data type.

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

## See also

[`nv_iota()`](https://r-xla.github.io/anvil/dev/reference/nv_iota.md)
for multi-dimensional sequences.

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
