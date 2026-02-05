# Concatenate

Concatenates tensors along a dimension. Operands are promoted to a
common data type and scalars are broadcast before concatenation.

## Usage

``` r
nv_concatenate(..., dimension = NULL)
```

## Arguments

- ...:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensors to concatenate. Must have the same shape except along
  `dimension`.

- dimension:

  (`integer(1)` \| `NULL`)  
  Dimension along which to concatenate. If `NULL` (default), assumes all
  inputs are at most 1-D and concatenates along dimension 1.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the common data type and a shape matching the inputs in all
dimensions except `dimension`, which is the sum of input sizes.

## See also

[`nvl_concatenate()`](https://r-xla.github.io/anvil/reference/nvl_concatenate.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 2, 3))
  y <- nv_tensor(c(4, 5, 6))
  nv_concatenate(x, y)
})
#> AnvilTensor
#>  1
#>  2
#>  3
#>  4
#>  5
#>  6
#> [ CPUf32{6} ] 
```
