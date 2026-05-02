# Variance Reduction

Computes the variance along the specified dimensions.

## Usage

``` r
nv_var(operand, dims, drop = TRUE, correction = 1L)
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

- correction:

  (`integer(1)`)  
  Degrees of freedom correction. Default is `1` (Bessel's correction).

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type as the input. When `drop = TRUE`, the reduced
dimensions are removed. When `drop = FALSE`, the reduced dimensions are
set to 1.

## Details

Uses Bessel's correction by default (`correction = 1`), matching R's
[`var()`](https://rdrr.io/r/stats/cor.html). Set `correction = 0` for
population variance.

## See also

[`nv_sd()`](https://r-xla.github.io/anvl/dev/reference/nv_sd.md),
[`nv_mean()`](https://r-xla.github.io/anvl/dev/reference/nv_mean.md)

## Examples

``` r
x <- nv_array(c(1, 2, 3, 4, 5))
nv_var(x, dims = 1L)
#> AnvlArray
#>  2.5000
#> [ CPUf32{} ] 
```
