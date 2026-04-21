# Variance Reduction

Computes the variance along the specified dimensions.

## Usage

``` r
nv_var(operand, dims, drop = TRUE, correction = 1L)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

- dims:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimensions to reduce.

- drop:

  (`logical(1)`)  
  Whether to drop reduced dimensions.

- correction:

  (`integer(1)`)  
  Degrees of freedom correction. Default is `1` (Bessel's correction).

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same data type as the input. When `drop = TRUE`, the reduced
dimensions are removed. When `drop = FALSE`, the reduced dimensions are
set to 1.

## Details

Uses Bessel's correction by default (`correction = 1`), matching R's
[`var()`](https://rdrr.io/r/stats/cor.html). Set `correction = 0` for
population variance.

## See also

[`nv_sd()`](https://r-xla.github.io/anvil/dev/reference/nv_sd.md),
[`nv_reduce_mean()`](https://r-xla.github.io/anvil/dev/reference/nv_reduce_mean.md)

## Examples

``` r
x <- nv_array(c(1, 2, 3, 4, 5))
nv_var(x, dims = 1L)
#> AnvilArray
#>  2.5000
#> [ CPUf32{} ] 
```
