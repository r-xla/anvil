# Broadcast to Shape

Broadcasts a tensor to a target shape using NumPy-style broadcasting
rules.

## Usage

``` r
nv_broadcast_to(operand, shape)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Target shape. Each existing dimension must either match or be 1.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the given `shape` and the same data type as `operand`.

## See also

[`nv_broadcast_tensors()`](https://r-xla.github.io/anvil/reference/nv_broadcast_tensors.md),
[`nv_broadcast_scalars()`](https://r-xla.github.io/anvil/reference/nv_broadcast_scalars.md),
[`nvl_broadcast_in_dim()`](https://r-xla.github.io/anvil/reference/nvl_broadcast_in_dim.md)
for the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 2, 3))
  nv_broadcast_to(x, shape = c(2, 3))
})
#> AnvilTensor
#>  1 2 3
#>  1 2 3
#> [ CPUf32{2,3} ] 
```
