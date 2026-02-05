# Broadcast Scalars to Common Shape

Broadcast scalar tensors to match the shape of non-scalar tensors. All
non-scalar tensors must have the same shape.

## Usage

``` r
nv_broadcast_scalars(...)
```

## Arguments

- ...:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensors to broadcast. Scalars will be broadcast to the common
  non-scalar shape.

## Value

([`list()`](https://rdrr.io/r/base/list.html) of
[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
List of broadcasted tensors.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 2, 3))
  # scalar 1 is broadcast to shape [3]
  nv_broadcast_scalars(x, 1)
})
#> [[1]]
#> AnvilTensor
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
#> 
#> [[2]]
#> AnvilTensor
#>  1
#>  1
#>  1
#> [ CPUf32?{3} ] 
#> 
```
