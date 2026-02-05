# Primitive Reshape

Reshapes a tensor to a new shape without changing the underlying data.
Note that row-major order is used, which differs from R's column-major
order.

## Usage

``` r
nvl_reshape(operand, shape)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of any data type.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Target shape. Must have the same number of elements as `operand`.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same data type as the input and the given `shape`. It is
ambiguous if the input is ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_reshape()`](https://r-xla.github.io/stablehlo/reference/hlo_reshape.html).

## See also

[`nv_reshape()`](https://r-xla.github.io/anvil/reference/nv_reshape.md)

## Examples

``` r
jit_eval({
  x <- nv_tensor(1:6)
  nvl_reshape(x, shape = c(2, 3))
})
#> AnvilTensor
#>  1 2 3
#>  4 5 6
#> [ CPUi32{2,3} ] 
```
