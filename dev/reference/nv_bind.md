# Combine arrays by rows or columns

Combine arrays along the row (`nv_rbind`) or column (`nv_cbind`)
dimension. Arguments are first promoted to a common data type (see
[`nv_promote_to_common()`](https://r-xla.github.io/anvl/dev/reference/nv_promote_to_common.md)).

Each input is then handled according to its rank:

- **Rank 0 (scalar, shape
  [`integer()`](https://rdrr.io/r/base/integer.html)):** broadcast to
  match the non-stacked dimensions of the other inputs, with size 1
  along the stacked dimension. If every input is a scalar, scalars
  become `c(1, 1)` (so `nv_rbind(s1, s2)` returns a `c(2, 1)` matrix and
  `nv_cbind(s1, s2)` returns a `c(1, 2)` matrix).

- **Rank 1 (vector):** treated as a single row by `nv_rbind` (reshaped
  to `c(1, length(x))`) or a single column by `nv_cbind` (reshaped to
  `c(length(x), 1)`).

- **Rank \>= 2:** used as-is. Inputs are concatenated along dimension 1
  (`nv_rbind`) or dimension 2 (`nv_cbind`); their other dimensions must
  match.

Only scalars are broadcast; non-scalar inputs must already have
compatible shapes.

## Usage

``` r
nv_rbind(...)

nv_cbind(...)

# S3 method for class 'AnvlBox'
rbind(..., deparse.level = 1)

# S3 method for class 'AnvlArray'
rbind(..., deparse.level = 1)

# S3 method for class 'AnvlBox'
cbind(..., deparse.level = 1)

# S3 method for class 'AnvlArray'
cbind(..., deparse.level = 1)
```

## Arguments

- ...:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Arrays to combine. Inputs are promoted to a common data type.

- deparse.level:

  Ignored. Kept for compatibility with
  [`base::rbind()`](https://rdrr.io/r/base/cbind.html) and
  [`base::cbind()`](https://rdrr.io/r/base/cbind.html).

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
An array of rank `max(2, max_rank(inputs))`. The stacked dimension has
size equal to the sum of stacked-dimension sizes (counting scalars as
1); the other dimensions match the non-scalar inputs.

## Differences from base R

[`base::rbind()`](https://rdrr.io/r/base/cbind.html) and
[`base::cbind()`](https://rdrr.io/r/base/cbind.html) applied to an
[`array()`](https://rdrr.io/r/base/array.html) of rank \> 2 flatten the
trailing dimensions into the column axis (so a `c(2, 3, 4)` array
becomes a `2 x 12` matrix). `nv_rbind` and `nv_cbind` instead preserve
all non-stacked dimensions: combining two `c(2, 3, 4)` arrays with
`nv_rbind` produces a `c(4, 3, 4)` array, and with `nv_cbind` a
`c(2, 6, 4)` array.

Base R also recycles shorter inputs to fill the result; `nv_rbind` and
`nv_cbind` only broadcast scalars, never longer non-scalar inputs.

## See also

[`nv_concatenate()`](https://r-xla.github.io/anvl/dev/reference/nv_concatenate.md)
for concatenation along an arbitrary dimension without the rank-1
row/column reshape.

## Examples

``` r
# Vectors as rows / columns
nv_rbind(nv_array(1:3), nv_array(4:6))
#> AnvlArray
#>  1 2 3
#>  4 5 6
#> [ CPUi32{2,3} ] 
nv_cbind(nv_array(1:3), nv_array(4:6))
#> AnvlArray
#>  1 4
#>  2 5
#>  3 6
#> [ CPUi32{3,2} ] 

# Scalar broadcasting
nv_rbind(nv_array(matrix(1:6, nrow = 2)), nv_scalar(0))
#> AnvlArray
#>  1 3 5
#>  2 4 6
#>  0 0 0
#> [ CPUf32{3,3} ] 

# Rank-3 arrays preserve trailing dimensions
a <- nv_array(array(1:24, dim = c(2, 3, 4)))
shape(nv_rbind(a, a)) # c(4, 3, 4)
#> [1] 4 3 4
```
