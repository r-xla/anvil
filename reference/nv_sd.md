# Standard Deviation Reduction

Computes the standard deviation along the specified dimensions.

## Usage

``` r
nv_sd(operand, dims, drop = TRUE, correction = 1L)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
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

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same data type as the input. When `drop = TRUE`, the reduced
dimensions are removed. When `drop = FALSE`, the reduced dimensions are
set to 1.

## Details

Uses Bessel's correction by default (`correction = 1`), matching R's
[`sd()`](https://rdrr.io/r/stats/sd.html). Set `correction = 0` for
population standard deviation.

## See also

[`nv_var()`](https://r-xla.github.io/anvl/reference/nv_var.md),
[`nv_reduce_mean()`](https://r-xla.github.io/anvl/reference/nv_reduce_mean.md)

## Examples

``` r
x <- nv_array(c(1, 2, 3, 4, 5))
nv_sd(x, dims = 1L)
#> AnvlArray
#>  1.5811
#> [ CPUf32{} ] 
```
