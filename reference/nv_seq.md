# Sequence

Creates a tensor with values increasing from start to end.

## Usage

``` r
nv_seq(start, end, dtype = "i32", ambiguous = FALSE)
```

## Arguments

- start, end:

  (`integer(1)`)  
  Start and end values.

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the vignette "Type Promotion" for more details.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
