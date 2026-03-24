# Create a Debug Box

Convenience constructor that creates a
[`DebugBox`](https://r-xla.github.io/anvil/dev/reference/DebugBox.md)
from a data type and shape, without having to manually construct an
[`AbstractArray`](https://r-xla.github.io/anvil/dev/reference/AbstractArray.md)
first.

## Usage

``` r
debug_box(dtype, shape, ambiguous = FALSE)
```

## Arguments

- dtype:

  (`character(1)` \|
  [`tengen::DataType`](https://r-xla.github.io/tengen/reference/DataType.html))  
  Data type.

- shape:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Shape.

- ambiguous:

  (`logical(1)`)  
  Whether the type is ambiguous. Ambiguous types usually arise from R
  literals (e.g., `1L`, `1.0`) and follow special promotion rules. See
  the
  [`vignette("type-promotion")`](https://r-xla.github.io/anvil/dev/articles/type-promotion.md)
  for more details.

## Value

([`DebugBox`](https://r-xla.github.io/anvil/dev/reference/DebugBox.md))

## See also

[DebugBox](https://r-xla.github.io/anvil/dev/reference/DebugBox.md),
[AbstractArray](https://r-xla.github.io/anvil/dev/reference/AbstractArray.md),
[`trace_fn()`](https://r-xla.github.io/anvil/dev/reference/trace_fn.md)

## Examples

``` r
# Create a debug box representing a 2x3 f32 array
db <- debug_box("f32", c(2L, 3L))
db
#> f32{2,3}
dtype(db)
#> <f32>
shape(db)
#> [1] 2 3
```
