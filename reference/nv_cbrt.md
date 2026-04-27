# Cube Root

Element-wise cube root.

## Usage

``` r
nv_cbrt(operand)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md))  
  Operand.

## Value

[`arrayish`](https://r-xla.github.io/anvl/reference/arrayish.md)  
Has the same shape and data type as the input.

## See also

[`prim_cbrt()`](https://r-xla.github.io/anvl/reference/prim_cbrt.md) for
the underlying primitive.

## Examples

``` r
x <- nv_array(c(1, 8, 27))
nv_cbrt(x)
#> AnvlArray
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
