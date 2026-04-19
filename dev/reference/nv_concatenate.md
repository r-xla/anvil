# Concatenate

Concatenates arrays along a dimension. Operands are promoted to a common
data type and scalars are broadcast before concatenation.

## Usage

``` r
nv_concatenate(..., dimension = NULL)
```

## Arguments

- ...:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrays to concatenate. Must have the same shape except along
  `dimension`.

- dimension:

  (`integer(1)` \| `NULL`)  
  Dimension along which to concatenate. If `NULL` (default), assumes all
  inputs are at most 1-D and concatenates along dimension 1.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the common data type and a shape matching the inputs in all
dimensions except `dimension`, which is the sum of input sizes.

## See also

[`nvl_concatenate()`](https://r-xla.github.io/anvil/dev/reference/nvl_concatenate.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_array(c(1, 2, 3))
  y <- nv_array(c(4, 5, 6))
  nv_concatenate(x, y)
})
#> AnvilArray
#>  1
#>  2
#>  3
#>  4
#>  5
#>  6
#> [ CPUf32{6} ] 
```
