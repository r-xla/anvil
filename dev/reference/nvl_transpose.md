# Primitive Transpose

Permutes the dimensions of an array.

## Usage

``` r
nvl_transpose(operand, permutation)
```

## Arguments

- operand:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Arrayish value of any data type.

- permutation:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Specifies the new ordering of dimensions. Must be a permutation of
  `seq_len(ndims)` where `ndims` is the number of dimensions of
  `operand`.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)  
Has the same data type as the input and shape
`nv_shape(operand)[permutation]`. It is ambiguous if the input is
ambiguous.

## Implemented Rules

- `stablehlo`

- `quickr`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_transpose()`](https://r-xla.github.io/stablehlo/reference/hlo_transpose.html).

## See also

[`nv_transpose()`](https://r-xla.github.io/anvil/dev/reference/nv_transpose.md),
[`t()`](https://rdrr.io/r/base/t.html)

## Examples

``` r
jit_eval({
  x <- nv_array(matrix(1:6, nrow = 2))
  nvl_transpose(x, permutation = c(2L, 1L))
})
#> AnvilArray
#>  1 2
#>  3 4
#>  5 6
#> [ CPUi32{3,2} ] 
```
