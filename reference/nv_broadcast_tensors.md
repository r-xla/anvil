# Broadcast Tensors to a Common Shape

Broadcasts tensors to a common shape using NumPy-style broadcasting
rules.

## Usage

``` r
nv_broadcast_tensors(...)
```

## Arguments

- ...:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensors to broadcast.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
List of tensors, all with the same shape.

## Broadcasting Rules

1.  If the tensors have different numbers of dimensions, prepend size-1
    dimensions to the shorter shape.

2.  For each dimension: if the sizes match, keep them; if one is 1,
    expand it to the other's size; otherwise raise an error.

## See also

[`nv_broadcast_scalars()`](https://r-xla.github.io/anvil/reference/nv_broadcast_scalars.md),
[`nv_broadcast_to()`](https://r-xla.github.io/anvil/reference/nv_broadcast_to.md)

## Examples

``` r
jit_eval({
  x <- nv_tensor(matrix(1:6, nrow = 2))
  y <- nv_tensor(c(10, 20, 30))
  nv_broadcast_tensors(x, y)
})
#> [[1]]
#> AnvilTensor
#>  1 3 5
#>  2 4 6
#> [ CPUi32{2,3} ] 
#> 
#> [[2]]
#> AnvilTensor
#>  10 20 30
#>  10 20 30
#> [ CPUf32{2,3} ] 
#> 
```
