# Primitive Pad

Pads a tensor with a given padding value.

## Usage

``` r
nvl_pad(
  operand,
  padding_value,
  edge_padding_low,
  edge_padding_high,
  interior_padding
)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of any data type.

- padding_value:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Scalar value to use for padding. Must have the same dtype as
  `operand`.

- edge_padding_low:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Amount of padding to add at the start of each dimension.

- edge_padding_high:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Amount of padding to add at the end of each dimension.

- interior_padding:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Amount of padding to add between elements in each dimension.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same data type as `operand`. For the output shape see the
underlying stablehlo documentation
([`stablehlo::hlo_pad()`](https://r-xla.github.io/stablehlo/reference/hlo_pad.html)).
It is ambiguous if the input is ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_pad()`](https://r-xla.github.io/stablehlo/reference/hlo_pad.html).

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 2, 3))
  nvl_pad(x, nv_scalar(0),
    edge_padding_low = 2L, edge_padding_high = 1L, interior_padding = 0L
  )
})
#> AnvilTensor
#>  0
#>  0
#>  1
#>  2
#>  3
#>  0
#> [ CPUf32{6} ] 
```
