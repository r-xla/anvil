# Sequence

Creates a 1-D tensor with integer values from `start` to `end`
(inclusive), analogous to R's `seq(start, end)`.

## Usage

``` r
nv_seq(start, end, dtype = "i32", ambiguous = FALSE)
```

## Arguments

- start, end:

  (`integer(1)`)  
  Start and end values. Must satisfy `start <= end`.

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the
  [`vignette("type-promotion")`](https://r-xla.github.io/anvil/articles/type-promotion.md)
  for more details.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
1-D tensor of length `end - start + 1`.

## See also

[`nv_iota()`](https://r-xla.github.io/anvil/reference/nv_iota.md) for
multi-dimensional sequences.

## Examples

``` r
jit_eval(nv_seq(3, 7))
#> AnvilTensor
#>  3
#>  4
#>  5
#>  6
#>  7
#> [ CPUi32{5} ] 
```
