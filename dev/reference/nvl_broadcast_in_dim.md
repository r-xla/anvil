# Primitive Broadcast

Broadcasts an array to a new shape by replicating the data along new or
size-1 dimensions.

## Usage

``` r
nvl_broadcast_in_dim(operand, shape, broadcast_dimensions)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Target shape. Each mapped dimension must either match the
  corresponding operand dimension or the operand dimension must be 1.

- broadcast_dimensions:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Maps each dimension of `operand` to a dimension of the output. Must
  have length equal to the number of dimensions of `operand`.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same data type as the input and the given `shape`. It is
ambiguous if the input is ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `reverse`

## StableHLO

Lowers to
[`stablehlo::hlo_broadcast_in_dim()`](https://r-xla.github.io/stablehlo/reference/hlo_broadcast_in_dim.html).

## See also

[`nv_broadcast_to()`](https://r-xla.github.io/anvil/dev/reference/nv_broadcast_to.md)

## Examples

``` r
jit_eval({
  x <- nv_array(c(1, 2, 3))
  nvl_broadcast_in_dim(x, shape = c(2, 3), broadcast_dimensions = 2L)
})
#> AnvilArray
#>  1 2 3
#>  1 2 3
#> [ CPUf32{2,3} ] 
```
