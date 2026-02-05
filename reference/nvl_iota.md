# Primitive Iota

Creates a tensor with values increasing along the specified dimension.

## Usage

``` r
nvl_iota(dim, dtype, shape, start = 1L, ambiguous = FALSE)
```

## Arguments

- dim:

  (`integer(1)`)  
  Dimension along which values increase (1-indexed).

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape of the output tensor.

- start:

  (`integer(1)`)  
  Starting value.

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the
  [`vignette("type-promotion")`](https://r-xla.github.io/anvil/articles/type-promotion.md)
  for more details.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
Has the given `dtype` and `shape`.

## Implemented Rules

- `stablehlo`

## StableHLO

Lowers to
[`stablehlo::hlo_iota()`](https://r-xla.github.io/stablehlo/reference/hlo_iota.html).

## See also

[`nv_iota()`](https://r-xla.github.io/anvil/reference/nv_iota.md)

## Examples

``` r
jit_eval(nvl_iota(dim = 1L, dtype = "i32", shape = 5L))
#> AnvilTensor
#>  1
#>  2
#>  3
#>  4
#>  5
#> [ CPUi32{5} ] 
```
