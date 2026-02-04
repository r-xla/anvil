# Create a Shape object

Constructs a `Shape` representing tensor dimensions.

## Usage

``` r
Shape(dims = integer())
```

## Arguments

- dims:

  An [`integer()`](https://rdrr.io/r/base/integer.html) vector of
  dimension sizes (\>= 0).

## Value

A `Shape` object.

## See also

[`shape()`](https://r-xla.github.io/anvil/reference/shape.md),
[`stablehlo::Shape()`](https://r-xla.github.io/stablehlo/reference/Shape.html)

## Examples

``` r
Shape(c(2L, 3L))
#> (2x3)
```
