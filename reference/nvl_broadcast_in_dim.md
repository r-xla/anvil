# Primitive Broadcast

Broadcasts a tensor to a new shape.

## Usage

``` r
nvl_broadcast_in_dim(operand, shape, broadcast_dimensions)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Target shape.

- broadcast_dimensions:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Dimension mapping.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)

## Shapes

`length(broadcast_dimensions)` must equal the rank of `operand`. Each
dimension of `operand` must either be 1 or match
`shape[broadcast_dimensions[i]]`. Output shape is `shape`.

## StableHLO

Calls
[`stablehlo::hlo_broadcast_in_dim()`](https://r-xla.github.io/stablehlo/reference/hlo_broadcast_in_dim.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 2, 3))
  nvl_broadcast_in_dim(x, shape = c(2, 3), broadcast_dimensions = 2L)
})
#> AnvilTensor
#>  1 2 3
#>  1 2 3
#> [ CPUf32{2,3} ] 
```
