# Subset a Tensor

Extracts a subset from a tensor. You can also use the `[` operator.
Supports R-style indexing including scalar indices (which drop
dimensions), ranges (`a:b`), and
[`list()`](https://rdrr.io/r/base/list.html) for selecting multiple
elements along a dimension.

## Usage

``` r
# S3 method for class 'AnvilBox'
x[...]

# S3 method for class 'AnvilTensor'
x[...]

nv_subset(x, ...)
```

## Arguments

- x:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensor to subset.

- ...:

    
  Subset specifications, one per dimension. Omitted trailing dimensions
  select all elements. See
  [`vignette("subsetting")`](https://r-xla.github.io/anvil/articles/subsetting.md)
  for details.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)

## See also

[`nv_subset_assign()`](https://r-xla.github.io/anvil/reference/nv_subset_assign.md)
for updating subsets,
[`vignette("subsetting")`](https://r-xla.github.io/anvil/articles/subsetting.md)
for a comprehensive guide.

## Examples

``` r
jit_eval({
  x <- nv_tensor(matrix(1:12, nrow = 3))
  # Select row 2
  x[2, ]
})
#> AnvilTensor
#>   2
#>   5
#>   8
#>  11
#> [ CPUi32{4} ] 

jit_eval({
  x <- nv_tensor(matrix(1:12, nrow = 3))
  # Select rows 1 to 2, all columns
  x[1:2, ]
})
#> AnvilTensor
#>   1  4  7 10
#>   2  5  8 11
#> [ CPUi32{2,4} ] 
```
