# Primitive Convert

Converts the elements of a tensor to a different data type.

## Usage

``` r
nvl_convert(operand, dtype, ambiguous = FALSE)
```

## Arguments

- operand:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensorish value of any data type.

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Target data type.

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the
  [`vignette("type-promotion")`](https://r-xla.github.io/anvil/articles/type-promotion.md)
  for more details.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the given `dtype` and the same shape as `operand`. Ambiguity is
controlled by the `ambiguous` parameter.

## Implemented Rules

- `stablehlo`

- `backward`

## StableHLO

Lowers to
[`stablehlo::hlo_convert()`](https://r-xla.github.io/stablehlo/reference/hlo_convert.html).

## See also

[`nv_convert()`](https://r-xla.github.io/anvil/reference/nv_convert.md)

## Examples

``` r
jit_eval({
  x <- nv_tensor(c(1L, 2L, 3L))
  nvl_convert(x, dtype = "f32")
})
#> AnvilTensor
#>  1
#>  2
#>  3
#> [ CPUf32{3} ] 
```
