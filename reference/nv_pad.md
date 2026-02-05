# Pad

Pads a tensor with a given value at the edges and optionally between
elements.

## Usage

``` r
nv_pad(
  operand,
  padding_value,
  edge_padding_low,
  edge_padding_high,
  interior_padding = NULL
)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Operand.

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

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Amount of padding to add between elements in each dimension. If `NULL`
  (default), no interior padding is applied.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same data type as `operand`.

## See also

[`nvl_pad()`](https://r-xla.github.io/anvil/reference/nvl_pad.md) for
the underlying primitive.

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1, 2, 3))
  nv_pad(x, nv_scalar(0), edge_padding_low = 2L, edge_padding_high = 1L)
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
