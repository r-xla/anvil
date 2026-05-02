# Mean

Computes the arithmetic mean along the specified dimensions. You can
also use [`mean()`](https://rdrr.io/r/base/mean.html).

## Usage

``` r
nv_mean(operand, dims = NULL, drop = TRUE)

# S3 method for class 'AnvlBox'
mean(x, trim = 0, na.rm = FALSE, ...)

# S3 method for class 'AnvlArray'
mean(x, trim = 0, na.rm = FALSE, ...)
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

- x:

  ([`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md))  
  Operand.

- trim:

  Included for compatibility with the
  [`base::mean()`](https://rdrr.io/r/base/mean.html) generic. Only
  `trim = 0` is supported; passing any other value raises an error.

- na.rm:

  Included for compatibility with the
  [`base::mean()`](https://rdrr.io/r/base/mean.html) generic. anvl
  arrays do not carry `NA`s; passing `na.rm = TRUE` raises an error.

- ...:

  Unused.

## Value

[`arrayish`](https://r-xla.github.io/anvl/dev/reference/arrayish.md)  
Has the same data type as the input. When `drop = TRUE`, the reduced
dimensions are removed. When `drop = FALSE`, the reduced dimensions are
set to 1.

## Details

Implemented as `nv_reduce_sum(operand, dims, drop) / n` where `n` is the
product of the reduced dimension sizes.

## See also

[`nv_reduce_sum()`](https://r-xla.github.io/anvl/dev/reference/nv_reduce_sum.md)

## Examples

``` r
x <- nv_array(matrix(1:6, nrow = 2))
nv_mean(x)            # all dims -> scalar
#> AnvlArray
#>  3.5000
#> [ CPUf32?{} ] 
nv_mean(x, dims = 1L)
#> AnvlArray
#>  1.5000
#>  3.5000
#>  5.5000
#> [ CPUf32?{3} ] 
```
