# Primitive Transpose

Permutes the dimensions of a tensor.

## Usage

``` r
nvl_transpose(operand, permutation)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of any data type.

- permutation:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Specifies the new ordering of dimensions. Must be a permutation of
  `seq_len(ndims)` where `ndims` is the number of dimensions of
  `operand`.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the same data type as the input and shape
`nv_shape(operand)[permutation]`. It is ambiguous if the input is
ambiguous.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_transpose()`](https://r-xla.github.io/stablehlo/reference/hlo_transpose.html).

## See also

[`nv_transpose()`](https://r-xla.github.io/anvil/reference/nv_transpose.md),
[`t()`](https://rdrr.io/r/base/t.html)

## Examples

``` r
jit_eval({
  x <- nv_tensor(matrix(1:6, nrow = 2))
  nvl_transpose(x, permutation = c(2L, 1L))
})
#> AnvilTensor
#>  1 2
#>  3 4
#>  5 6
#> [ CPUi32{3,2} ] 
```
