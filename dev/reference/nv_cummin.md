# Cumulative Minimum

Running minimum, optionally along a single dimension.

## Usage

``` r
nv_cummin(operand, dim = NULL, with_indices = FALSE)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Operand.

- dim:

  (`integer(1)` \| `NULL`)  
  Dimension along which to accumulate. If `NULL` (default), the input is
  first flattened to a 1-D array, like
  [`base::cummin()`](https://rdrr.io/r/base/cumsum.html).

- with_indices:

  (`logical(1)`)  
  If `FALSE` (default), returns the running-minimum array. If `TRUE`,
  returns `list(values = ..., indices = ...)` where `indices` is the
  1-based index of the last occurrence of the running minimum at each
  position (dtype `i32`, matching torch). When `dim = NULL`, indices
  refer to the flattened input.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)
(when `with_indices = FALSE`) or named list of two arrays (when
`with_indices = TRUE`).

## Relation to base R

Both `nv_cummin()` (with `dim = NULL`) and
[`base::cummin()`](https://rdrr.io/r/base/cumsum.html) flatten a
multi-dimensional input to 1-D before accumulating, but the flatten
order differs: anvl arrays are row-major (C order), so the flattened
sequence iterates the last dim fastest, whereas base R uses column-major
(Fortran) order. The two agree on 1-D inputs.

## See also

[`prim_cummin()`](https://r-xla.github.io/anvl/dev/reference/prim_cummin.md)
for the underlying primitive.

## Examples

``` r
x <- nv_array(matrix(c(3, 1, 4, 1, 5, 9), nrow = 2))
nv_cummin(x)
#> AnvlArray
#>  3
#>  3
#>  3
#>  1
#>  1
#>  1
#> [ CPUf32{6} ] 
nv_cummin(x, dim = 1L)
#> AnvlArray
#>  3 4 5
#>  1 1 5
#> [ CPUf32{2,3} ] 
nv_cummin(x, dim = 1L, with_indices = TRUE)
#> $values
#> AnvlArray
#>  3 4 5
#>  1 1 5
#> [ CPUf32{2,3} ] 
#> 
#> $indices
#> AnvlArray
#>  1 1 1
#>  2 2 1
#> [ CPUi32{2,3} ] 
#> 
```
