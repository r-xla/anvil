# Update Subset

Updates elements of a tensor at specified positions, returning a new
tensor. You can also use the `[<-` operator.

## Usage

``` r
# S3 method for class 'AnvilBox'
x[...] <- value

# S3 method for class 'AnvilTensor'
x[...] <- value

nv_subset_assign(x, ..., value)
```

## Arguments

- x:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Tensor to update.

- ...:

    
  Subset specifications, one per dimension. See
  [`vignette("subsetting")`](https://r-xla.github.io/anvil/articles/subsetting.md)
  for details.

- value:

  ([`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md))  
  Replacement values. Scalars are broadcast to the subset shape.
  Non-scalar values must match the subset shape.

## Value

[`tensorish`](https://r-xla.github.io/anvil/reference/tensorish.md)  
A new tensor with the same shape as `x` and the subset replaced.

## See also

[`nv_subset()`](https://r-xla.github.io/anvil/reference/nv_subset.md),
[`vignette("subsetting")`](https://r-xla.github.io/anvil/articles/subsetting.md)
for a comprehensive guide.

## Examples

``` r
jit_eval({
  x <- nv_tensor(matrix(1:12, nrow = 3))
  # Set row 1 to zeros
  x[1, ] <- 0L
  x
})
#> AnvilTensor
#>   0  0  0  0
#>   2  5  8 11
#>   3  6  9 12
#> [ CPUi32{3,4} ] 
```
