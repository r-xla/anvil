# Subset an Array

Extracts a subset from an array. You can also use the `[` operator.
Supports R-style indexing including scalar indices (which drop
dimensions), ranges (`a:b`), and `array(c(...))` for selecting multiple
elements along a dimension.

## Usage

``` r
# S3 method for class 'AnvilBox'
x[...]

# S3 method for class 'AnvilArray'
x[...]

nv_subset(x, ...)
```

## Arguments

- x:

  ([`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md))  
  Array to subset.

- ...:

  Subset specifications, one per dimension. Omitted trailing dimensions
  select all elements. See
  [`vignette("subsetting")`](https://r-xla.github.io/anvil/dev/articles/subsetting.md)
  for details.

## Value

[`arrayish`](https://r-xla.github.io/anvil/dev/reference/arrayish.md)

## See also

[`nv_subset_assign()`](https://r-xla.github.io/anvil/dev/reference/nv_subset_assign.md)
for updating subsets,
[`vignette("subsetting")`](https://r-xla.github.io/anvil/dev/articles/subsetting.md)
for a comprehensive guide.

## Examples

``` r
if (FALSE) { # pjrt::plugins_downloaded()
jit_eval({
  x <- nv_array(matrix(1:12, nrow = 3))
  # Select row 2
  x[2, ]
})

jit_eval({
  x <- nv_array(matrix(1:12, nrow = 3))
  # Select rows 1 to 2, all columns
  x[1:2, ]
})
}
```
