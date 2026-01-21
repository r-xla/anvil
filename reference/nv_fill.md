# Constant

Create a constant.

## Usage

``` r
nv_fill(value, shape, dtype = NULL, ambiguous = FALSE)
```

## Arguments

- value:

  (any)  
  Value.

- shape:

  (integer())  
  Shape.

- dtype:

  (character(1))  
  Data type.

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the vignette "Type Promotion" for more details.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)
