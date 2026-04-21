# Unsqueeze

Inserts a dimension of size 1 at the specified position.

## Usage

``` r
nv_unsqueeze(operand, dim)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Operand.

- dim:

  (`integer(1)`)  
  Position at which to insert the new dimension.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same data type as `operand` with an extra dimension of size 1.

## See also

[`nv_squeeze()`](https://r-xla.github.io/anvil/dev/reference/nv_squeeze.md),
[`nv_reshape()`](https://r-xla.github.io/anvil/dev/reference/nv_reshape.md)

## Examples

``` r
x <- nv_array(c(1, 2, 3))
nv_unsqueeze(x, dim = 1L)
#> AnvilArray
#>  1 2 3
#> [ CPUf32{1,3} ] 
```
