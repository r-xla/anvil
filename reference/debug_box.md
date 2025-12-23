# Create a Debug Box

Create a debug box.

## Usage

``` r
debug_box(dtype, shape, ambiguous = FALSE)
```

## Arguments

- dtype:

  (`character(1)` \|
  [`stablehlo::TensorDataType`](https://r-xla.github.io/stablehlo/reference/TensorDataType.html))  
  Data type.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the vignette "Type Promotion" for more details.

## Value

([`DebugBox`](DebugBox.md))
